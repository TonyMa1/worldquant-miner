# File: python/pre_consultant/alpha_generator.py
# Modified to use OpenRouter API instead of MoonshotAI or direct Google Gemini

import argparse
import requests # Make sure requests is installed
import json
import os
import re
from time import sleep
from requests.auth import HTTPBasicAuth # For WQ Auth
import time
import logging
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from typing import List, Dict, Optional

# Configure logger
logger = logging.getLogger(__name__)
# Ensure logging is configured in the main block

# --- RetryQueue Class (for WQ simulations) ---
class RetryQueue:
    def __init__(self, generator, max_retries=3, retry_delay=60):
        self.queue = Queue()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.generator = generator
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def add(self, alpha: str, retry_count: int = 0):
        self.queue.put((alpha, retry_count))

    def _process_queue(self):
        while True:
            if not self.queue.empty():
                alpha, retry_count = self.queue.get()
                if retry_count >= self.max_retries:
                    logging.error(f"Max WQ retries exceeded for alpha: {alpha}")
                    continue

                try:
                    # Attempt WQ simulation submission
                    result = self.generator._test_alpha_impl(alpha)
                    if result.get("status") == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
                        logging.info(f"WQ Simulation limit exceeded, requeueing alpha: {alpha}")
                        time.sleep(self.retry_delay * (retry_count + 1)) # Exponential backoff maybe
                        self.add(alpha, retry_count + 1) # Re-add with incremented count
                    else:
                        # Track WQ submission attempts/results if needed, separate from self.generator.results
                        logging.debug(f"Processed alpha from WQ retry queue: {alpha}, Result status: {result.get('status')}")

                except Exception as e:
                    logging.error(f"Error processing alpha {alpha} from WQ retry queue: {str(e)}")

            time.sleep(1)

# --- AlphaGenerator Class ---
class AlphaGenerator:
    def __init__(self, credentials_path: str):
        self.sess = requests.Session()
        self.credentials_path = credentials_path
        self.setup_auth(credentials_path)
        self.results = [] # Stores completed WQ simulation results/details
        self.pending_results = {} # Stores pending WQ simulations {progress_url: {info}}
        self.retry_queue = RetryQueue(self) # For retrying WQ simulations if limit hit
        self.executor = ThreadPoolExecutor(max_workers=4) # For concurrent WQ simulations/checks
        self._hit_token_limit = False # Flag for potential AI token limits

    def setup_auth(self, credentials_path: str) -> None:
        """Set up authentication with WorldQuant Brain."""
        logging.info(f"Loading credentials from {credentials_path}")
        try:
            with open(credentials_path) as f:
                credentials = json.load(f)
        except FileNotFoundError:
             logging.error(f"Credentials file not found at {credentials_path}")
             raise
        except json.JSONDecodeError:
             logging.error(f"Could not decode JSON from credentials file: {credentials_path}")
             raise

        if not isinstance(credentials, list) or len(credentials) != 2:
            logging.error("Credentials file should contain a JSON list with [username, password]")
            raise ValueError("Invalid credentials format")

        username, password = credentials
        self.sess.auth = HTTPBasicAuth(username, password)

        logging.info("Authenticating with WorldQuant Brain...")
        try:
            response = self.sess.post('https://api.worldquantbrain.com/authentication', timeout=15)
            logging.info(f"WQ Authentication response status: {response.status_code}")
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
             logging.error(f"WorldQuant Brain authentication failed: {e}")
             raise Exception(f"Authentication failed: {e}") from e
        logging.info("WQ Authentication successful")


    def get_data_fields(self) -> List[Dict]:
        """Fetch available data fields from WorldQuant Brain."""
        # --- Keeping this function as is ---
        datasets = ['fundamental6', 'fundamental2', 'analyst4', 'model16', 'model51', 'news12']
        all_fields = []
        base_params = {'delay': 1, 'instrumentType': 'EQUITY', 'limit': 20, 'region': 'USA', 'universe': 'TOP3000'}
        print("Requesting data fields from multiple datasets...")
        try:
            for dataset in datasets:
                params = base_params.copy()
                params['dataset.id'] = dataset
                params['limit'] = 1
                print(f"Getting field count for dataset: {dataset}")
                count_response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params, timeout=15)
                if count_response.status_code == 200:
                    count_data = count_response.json()
                    total_fields = count_data.get('count', 0)
                    print(f"Total fields in {dataset}: {total_fields}")
                    if total_fields > 0:
                        fetch_limit = min(20, total_fields)
                        max_offset = max(0, total_fields - fetch_limit)
                        random_offset = random.randint(0, max_offset) if max_offset > 0 else 0
                        params['offset'] = random_offset
                        params['limit'] = fetch_limit
                        print(f"Fetching fields for {dataset} with offset {random_offset}")
                        response = self.sess.get('https://api.worldquantbrain.com/data-fields', params=params, timeout=15)
                        if response.status_code == 200:
                            data = response.json()
                            fields = data.get('results', [])
                            print(f"Found {len(fields)} fields in {dataset}")
                            all_fields.extend(fields)
                        else:
                            print(f"Failed to fetch fields for {dataset}: {response.text[:200]}")
                else:
                    print(f"Failed to get count for {dataset}: {count_response.text[:200]}")
            unique_fields_dict = {field['id']: field for field in all_fields if isinstance(field, dict) and 'id' in field}
            unique_fields = list(unique_fields_dict.values())
            print(f"Total unique WQ fields found: {len(unique_fields)}")
            return unique_fields
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch WQ data fields: {e}")
            return []
        except Exception as e:
             logger.error(f"Unexpected error fetching WQ data fields: {e}")
             return []

    def get_operators(self) -> List[Dict]:
        """Fetch available operators from WorldQuant Brain."""
        # --- Keeping this function as is ---
        print("Requesting WQ operators...")
        try:
            response = self.sess.get('https://api.worldquantbrain.com/operators', timeout=20)
            print(f"WQ Operators response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                print(f"Successfully fetched {len(data)} WQ operators.")
                return data
            elif isinstance(data, dict) and 'results' in data:
                 print(f"Successfully fetched {len(data['results'])} WQ operators.")
                 return data['results']
            else:
                 raise Exception(f"Unexpected WQ operators response format.")
        except requests.exceptions.RequestException as e:
             logger.error(f"Failed to get WQ operators: {e}")
             raise Exception(f"Failed to get WQ operators: {e}") from e
        except json.JSONDecodeError as e:
             logger.error(f"Failed to parse WQ operators JSON response: {e}")
             raise Exception(f"Failed to parse WQ operators JSON response: {e}") from e


    def clean_alpha_ideas(self, ideas: List[str]) -> List[str]:
        """Clean and validate alpha ideas, keeping only valid expressions."""
        # --- Keeping this function as is ---
        cleaned_ideas = []
        if not ideas:
             return cleaned_ideas
        for idea in ideas:
            idea_stripped = idea.strip()
            if not idea_stripped: continue
            if re.match(r'^\d+(\.\d+)?$|^[a-zA-Z_]+$', idea_stripped):
                logging.debug(f"Skipping potentially invalid expression (number/word): {idea_stripped}")
                continue
            common_words = ['example', 'alpha', 'this', 'that', 'the', 'is', 'are', 'use', 'uses', 'using', 'and', 'for', 'with', 'captures', 'provides', 'measures', 'factor', 'based', 'idea']
            if sum(1 for word in common_words if word in idea_stripped.lower().split()) > 3:
                 logging.debug(f"Skipping potentially descriptive text: {idea_stripped}")
                 continue
            if not re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(', idea_stripped):
                 logging.debug(f"Skipping expression lacking function calls: {idea_stripped}")
                 continue
            cleaned_ideas.append(idea_stripped)
        return cleaned_ideas

    # --- MODIFIED Function to use OpenRouter ---
    def generate_alpha_ideas(self, data_fields: List[Dict], operators: List[Dict]) -> List[str]:
        """Generate alpha ideas using OpenRouter API."""
        print("Organizing operators by category...")
        operator_by_category = {}
        if operators:
            for op in operators:
                category = op.get('category', 'Uncategorized')
                if category not in operator_by_category:
                    operator_by_category[category] = []
                operator_by_category[category].append({
                    'name': op.get('name', 'N/A'),
                    'type': op.get('type', 'SCALAR'),
                    'definition': op.get('definition', 'N/A'),
                    'description': op.get('description', 'N/A')
                })
        else:
             logging.warning("No WQ operators data available for the prompt.")

        try:
            if self._hit_token_limit: # Check flag from previous attempts
                logging.info("Clearing potentially problematic results due to previous token limit hit.")
                # Decide if clearing self.results is appropriate here, might lose WQ results
                # self.results = []
                self._hit_token_limit = False # Reset flag

            sampled_operators = {}
            for category, ops in operator_by_category.items():
                 if ops:
                    sample_size = max(1, int(len(ops) * 0.5))
                    sampled_operators[category] = random.sample(ops, min(sample_size, len(ops)))

            print("Preparing prompt for OpenRouter...")
            def format_operators(ops):
                formatted = []
                for op in ops:
                    formatted.append(f"{op['name']} ({op['type']})\n"
                                   f"  Definition: {op['definition']}\n"
                                   f"  Description: {op['description']}")
                return formatted

            prompt_sections = []
            for category, ops in sampled_operators.items():
                if ops:
                    prompt_sections.append(f"{category}:\n{chr(10).join(format_operators(ops))}")

            # Construct the prompt
            prompt = f"""Generate 5 unique alpha factor expressions using the available operators and data fields for the WorldQuant Brain platform (FASTEXPR language). Return ONLY the expressions, one per line, with no comments, explanations, or markdown formatting like backticks.

Available Data Fields (sample):
{[field.get('id', 'N/A') for field in data_fields[:30]]}

Available Operators by Category (sample):
{chr(10).join(prompt_sections)}

Requirements:
1. Create potentially profitable alpha factors.
2. Use the provided operators and data fields, respecting operator types (SCALAR, VECTOR, MATRIX).
3. Combine multiple operators (ts_, rank, zscore, arithmetic, logical, vector, group, etc.).
4. Ensure expressions are syntactically plausible for FASTEXPR.
5. Aim for diversity.

Tips:
- Common fields: 'open', 'high', 'low', 'close', 'volume', 'returns', 'vwap', 'cap'.
- Use 'rank' or 'zscore' for normalization.
- Use time series operators like 'ts_mean', 'ts_std_dev', 'ts_rank', 'ts_delta' with lookback windows (e.g., 5, 10, 20, 60).
- Use 'ts_corr' or 'ts_covariance'.

Example of desired output format (do not include this specific example):
-ts_corr(rank(close), rank(volume), 20)

Generate 5 distinct FASTEXPR expressions now:
"""
            # --- START: OpenRouter API Integration ---
            print("Configuring OpenRouter request...")
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it and restart.")

            your_site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost") # Optional site URL
            your_site_name = os.getenv("OPENROUTER_SITE_NAME", "WQAlphaGen") # Optional site name

            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": your_site_url,
                "X-Title": your_site_name,
            }

            # Verify model identifier on OpenRouter website - this might change
            openrouter_model = "google/gemini-pro" # Or "google/gemini-flash-1.5" or other available model

            data = json.dumps({
                "model": openrouter_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.45, # Slightly higher temp for more diverse ideas
                # "max_tokens": 1024, # Optional: Limit response length
            })

            print(f"Sending request to OpenRouter API (Model: {openrouter_model})...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=data,
                timeout=90 # Increased timeout for potentially longer AI generation
            )

            print(f"OpenRouter API response status: {response.status_code}")
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()

            # Extract content
            if not response_data.get("choices"):
                 error_detail = response_data.get("error", {}).get("message", "No choices found in response.")
                 # Check for token limit messages specifically if possible
                 if "context_length_exceeded" in error_detail.lower():
                      self._hit_token_limit = True
                      logging.warning("OpenRouter Error: Token limit likely exceeded. Prompt may be too long.")
                 raise Exception(f"OpenRouter API response missing 'choices'. Error detail: {error_detail}")

            content = response_data["choices"][0]["message"]["content"]
            # --- END: OpenRouter API Integration ---

            logging.debug(f"Raw OpenRouter response content:\n{content}")

            # Parse and clean the response
            alpha_ideas = []
            processed_lines = set()
            for line in content.split('\n'):
                line = line.strip()
                # Remove potential markdown formatting or unwanted prefixes/suffixes
                line = re.sub(r'^```(fast)?expr\s*', '', line, flags=re.IGNORECASE | re.MULTILINE)
                line = re.sub(r'```$', '', line).strip()
                line = re.sub(r'^\d+\.\s*', '', line) # Remove leading numbering
                line = re.sub(r'^-?\s*', '', line) # Remove leading '-' or spaces sometimes added

                # Filter out explanations or non-expressions
                if not line or line.lower().startswith(('sure', 'here are', 'alpha', '#', '*', 'comment', 'note', 'explanation')):
                    continue

                # Add if looks plausible and not duplicate
                if '(' in line and ')' in line and line not in processed_lines:
                    alpha_ideas.append(line)
                    processed_lines.add(line)

            print(f"Generated {len(alpha_ideas)} alpha ideas (before cleaning)")
            if alpha_ideas:
                 for i, alpha in enumerate(alpha_ideas, 1):
                      print(f"Raw Alpha {i}: {alpha[:100]}...") # Print truncated

            # Clean ideas further
            cleaned_ideas = self.clean_alpha_ideas(alpha_ideas)
            logging.info(f"Found {len(cleaned_ideas)} plausible alpha expressions after cleaning.")

            return cleaned_ideas

        except requests.exceptions.Timeout:
             logging.error("Timeout connecting to OpenRouter API.")
             return []
        except requests.exceptions.RequestException as e:
             status_code = e.response.status_code if e.response is not None else "N/A"
             response_text = e.response.text[:500] if e.response is not None else "N/A"
             logging.error(f"OpenRouter API request failed (Status: {status_code}): {e}. Response: {response_text}")
             if status_code == 401: logging.error("Check your OPENROUTER_API_KEY.")
             elif status_code == 402: logging.error("Check your OpenRouter account credits.")
             elif status_code == 429: logging.error("OpenRouter rate limit exceeded.")
             elif status_code == 400 and "context_length_exceeded" in response_text.lower():
                  self._hit_token_limit = True
                  logging.error("OpenRouter Error: Token limit likely exceeded.")
             return []
        except Exception as e:
            logging.error(f"Error generating alpha ideas with OpenRouter: {str(e)}", exc_info=True)
            return []

    # --- WorldQuant Interaction Functions (Keep As Is or with minor refinements) ---

    def test_alpha_batch(self, alphas: List[str]) -> int:
        """Submit a batch of alphas to WorldQuant Brain for testing with monitoring."""
        if not alphas:
             logging.info("No alphas provided to test_alpha_batch.")
             return 0
        logging.info(f"Starting WQ batch submission/test of {len(alphas)} alphas")
        submitted_count = 0
        queued_for_retry = 0
        # Submit all alphas to WQ, handle immediate errors/limits
        for i, alpha in enumerate(alphas, 1):
            logging.info(f"Submitting alpha {i}/{len(alphas)} to WQ: {alpha[:80]}...")
            try:
                 result = self._test_alpha_impl(alpha)
                 if result.get("status") == "success" and result.get("result", {}).get("progress_url"):
                      progress_url = result["result"]["progress_url"]
                      sim_id_placeholder = progress_url.split('/')[-1]
                      self.pending_results[progress_url] = {"alpha": alpha, "sim_id": sim_id_placeholder, "status": "pending", "attempts": 0}
                      submitted_count += 1
                      logging.info(f"Successfully submitted '{alpha[:80]}...' to WQ. URL: {progress_url}")
                 elif result.get("status") == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
                      self.retry_queue.add(alpha)
                      queued_for_retry += 1
                      logging.warning(f"Queued '{alpha[:80]}...' for WQ retry due to limit.")
                 else:
                      logging.error(f"Failed to submit alpha '{alpha[:80]}...' to WQ: {result.get('message', 'Unknown error')}")
            except Exception as e:
                logging.error(f"Unexpected error submitting alpha '{alpha[:80]}...' to WQ: {str(e)}")
            sleep(0.5)
        logging.info(f"WQ Batch submission phase complete: {submitted_count} submitted, {queued_for_retry} queued for retry")

        # Monitor progress
        total_successful_simulations = 0
        monitoring_start_time = time.time()
        MONITORING_TIMEOUT_SECONDS = 60 * 15 # 15 minutes
        while self.pending_results and (time.time() - monitoring_start_time) < MONITORING_TIMEOUT_SECONDS:
            logging.info(f"Monitoring {len(self.pending_results)} pending WQ simulations...")
            try:
                successful_in_check = self.check_pending_results()
                total_successful_simulations += successful_in_check
                if successful_in_check > 0: logging.info(f"{successful_in_check} WQ simulations completed successfully in this check.")
                elif not self.pending_results: break
                else: logging.info("No WQ simulations completed in this check.")
            except Exception as monitor_err:
                 logging.error(f"Error during WQ monitoring loop: {monitor_err}")
            sleep(15)
        if self.pending_results:
             logging.warning(f"WQ Monitoring timed out or stopped. {len(self.pending_results)} simulations still pending.")
        logging.info(f"WQ Batch monitoring complete: {total_successful_simulations} successful simulations found in this batch.")
        return total_successful_simulations


    def check_pending_results(self) -> int:
        """Check status of all pending WQ simulations. Returns count of newly successful."""
        completed_urls = []
        successful_count = 0
        max_check_attempts = 5
        urls_to_check = list(self.pending_results.keys()) # Iterate over copy

        for progress_url in urls_to_check:
            if progress_url not in self.pending_results: continue
            info = self.pending_results[progress_url]
            alpha_expr = info["alpha"]
            check_attempt = info.get("attempts", 0) + 1
            self.pending_results[progress_url]["attempts"] = check_attempt

            if check_attempt > max_check_attempts:
                logging.error(f"Max check attempts reached for WQ sim {progress_url} ('{alpha_expr[:50]}...'). Removing.")
                completed_urls.append(progress_url)
                continue
            try:
                logging.debug(f"Checking WQ simulation {progress_url} for alpha: {alpha_expr[:50]}...")
                sim_progress_resp = self.sess.get(progress_url, timeout=20)
                if sim_progress_resp.status_code == 429:
                    retry_after = int(sim_progress_resp.headers.get('Retry-After', 5))
                    logging.warning(f"WQ rate limit hit checking status. Waiting {retry_after}s...")
                    sleep(retry_after)
                    continue
                if sim_progress_resp.status_code == 401:
                    logging.warning("WQ Auth expired checking status, refreshing...")
                    self.setup_auth(self.credentials_path)
                    continue
                sim_progress_resp.raise_for_status()
                if not sim_progress_resp.text.strip():
                    logging.debug(f"WQ Simulation {progress_url} still initializing...")
                    continue
                try:
                    sim_result = sim_progress_resp.json()
                except json.JSONDecodeError:
                     logging.warning(f"WQ Simulation {progress_url} response not JSON yet: {sim_progress_resp.text[:100]}...")
                     continue
                status = sim_result.get("status")
                sim_id = sim_result.get("id", info["sim_id"])
                self.pending_results[progress_url]["sim_id"] = sim_id
                logging.info(f"WQ Simulation {sim_id} ({alpha_expr[:50]}...) status: {status}")
                if status == "COMPLETE":
                    alpha_id_from_sim = sim_result.get("alpha")
                    if alpha_id_from_sim:
                        try:
                            alpha_resp = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id_from_sim}', timeout=20)
                            if alpha_resp.status_code == 200:
                                alpha_data = alpha_resp.json()
                                self.results.append({"alpha_expression": alpha_expr, "simulation_result": sim_result, "alpha_details": alpha_data})
                                fitness = alpha_data.get("is", {}).get("fitness", 0)
                                sharpe = alpha_data.get("is", {}).get("sharpe", 0)
                                logging.info(f"WQ Alpha {alpha_id_from_sim} completed. Fitness: {fitness:.2f}, Sharpe: {sharpe:.2f}")
                                if fitness > 0.5 and sharpe > 0.8:
                                    logging.info(f"*** Found promising WQ alpha! *** Fitness: {fitness:.2f}, Sharpe: {sharpe:.2f}")
                                    self.log_hopeful_alpha(alpha_expr, alpha_data)
                                successful_count += 1
                            else:
                                logging.error(f"Failed to fetch details for completed alpha {alpha_id_from_sim}. Status: {alpha_resp.status_code}")
                                self.results.append({"alpha_expression": alpha_expr, "simulation_result": sim_result, "alpha_details": {"error": f"Failed fetch, status {alpha_resp.status_code}"}})
                        except Exception as alpha_fetch_err:
                             logging.error(f"Error fetching details for alpha {alpha_id_from_sim}: {alpha_fetch_err}")
                             self.results.append({"alpha_expression": alpha_expr, "simulation_result": sim_result, "alpha_details": {"error": f"Exception fetching: {alpha_fetch_err}"}})
                    else:
                         logging.warning(f"WQ Simulation {sim_id} complete but no alpha ID.")
                         self.results.append({"alpha_expression": alpha_expr, "simulation_result": sim_result, "alpha_details": None})
                    completed_urls.append(progress_url)
                elif status in ["FAILED", "ERROR"]:
                    error_message = sim_result.get("message", "Unknown error")
                    logging.error(f"WQ Simulation {sim_id} failed for '{alpha_expr[:50]}...': {error_message}")
                    self.results.append({"alpha_expression": alpha_expr, "simulation_result": sim_result, "alpha_details": None, "status": "failed"})
                    completed_urls.append(progress_url)
                elif status == "RUNNING":
                     logging.debug(f"WQ Simulation {sim_id} still running...")
                else:
                     logging.warning(f"Unknown WQ simulation status '{status}' for {sim_id}.")
            except requests.exceptions.RequestException as req_err:
                 logging.error(f"Network error checking WQ result for {progress_url}: {req_err}")
            except Exception as e:
                 logging.error(f"Error checking WQ result for {progress_url} ('{alpha_expr[:50]}...'): {str(e)}")
                 # Decide if unexpected errors should mark as complete to avoid loops
                 # completed_urls.append(progress_url)
        # Cleanup completed
        for url in completed_urls:
             if url in self.pending_results:
                 del self.pending_results[url]
        return successful_count


    def test_alpha(self, alpha: str) -> Dict:
        """Submits alpha to WQ and potentially adds to WQ retry queue."""
        # --- Keeping this function as is ---
        result = self._test_alpha_impl(alpha)
        if result.get("status") == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
            self.retry_queue.add(alpha)
            logging.warning(f"WQ simulation limit hit for '{alpha[:80]}...'. Added to retry queue.")
            return {"status": "queued", "message": "Added to WQ retry queue"}
        return result

    def _test_alpha_impl(self, alpha_expression: str) -> Dict:
        """Implementation of alpha testing (WorldQuant simulation submission)."""
        # --- Keeping this function as is, with minor logging/error improvements ---
        logging.debug(f"Preparing WQ submission: {alpha_expression[:100]}...")
        def submit_simulation():
            simulation_data = {
                'type': 'REGULAR',
                'settings': {'instrumentType': 'EQUITY', 'region': 'USA', 'universe': 'TOP3000', 'delay': 1, 'decay': 0, 'neutralization': 'INDUSTRY', 'truncation': 0.08, 'pasteurization': 'ON', 'unitHandling': 'VERIFY', 'nanHandling': 'OFF', 'language': 'FASTEXPR', 'visualization': False,},
                'regular': alpha_expression
            }
            return self.sess.post('https://api.worldquantbrain.com/simulations', json=simulation_data, timeout=30)
        try:
            sim_resp = submit_simulation()
            if sim_resp.status_code == 401:
                logging.warning("WQ Auth expired on submission, refreshing...")
                self.setup_auth(self.credentials_path)
                sim_resp = submit_simulation()
            if sim_resp.status_code == 400 and "invalid expression" in sim_resp.text.lower():
                 logging.error(f"WQ rejected invalid expression: {alpha_expression}. Error: {sim_resp.text[:200]}")
                 return {"status": "error", "message": f"Invalid Expression: {sim_resp.text}"}
            if sim_resp.status_code == 429:
                 retry_after = int(sim_resp.headers.get('Retry-After', 5))
                 logging.warning(f"WQ rate limit on submission. Wait {retry_after}s.")
                 return {"status": "error", "message": f"SIMULATION_LIMIT_EXCEEDED (Rate Limit Wait {retry_after}s)"}
            if sim_resp.status_code != 201:
                error_msg = f"WQ submission failed ({sim_resp.status_code}): {sim_resp.text[:500]}"
                logging.error(f"WQ submission failed for '{alpha_expression[:80]}...'. {error_msg}")
                if "simulation limit" in sim_resp.text.lower():
                     return {"status": "error", "message": f"SIMULATION_LIMIT_EXCEEDED: {sim_resp.text}"}
                else:
                     return {"status": "error", "message": error_msg}
            sim_progress_url = sim_resp.headers.get('location')
            if not sim_progress_url:
                logging.error(f"WQ submission 201 but no progress URL for '{alpha_expression[:80]}...'.")
                return {"status": "error", "message": "No progress URL despite 201 status"}
            sim_id_placeholder = sim_progress_url.split('/')[-1]
            return {"status": "success", "result": {"id": sim_id_placeholder, "progress_url": sim_progress_url}}
        except requests.exceptions.Timeout:
             logging.error(f"Timeout submitting WQ sim: {alpha_expression[:80]}...")
             return {"status": "error", "message": "Request Timeout during submission"}
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error submitting WQ sim: {alpha_expression[:80]}...: {str(e)}")
            return {"status": "error", "message": f"Network Error: {str(e)}"}
        except Exception as e:
            logging.error(f"Unexpected error testing WQ alpha {alpha_expression[:80]}...: {str(e)}")
            return {"status": "error", "message": f"Unexpected Error: {str(e)}"}


    def log_hopeful_alpha(self, expression: str, alpha_data: Dict) -> None:
        """Log promising alphas (based on WQ results) to a JSON file."""
        # --- Keeping this function as is ---
        log_file = 'hopeful_alphas_wq.json' # Distinct filename
        existing_data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f: existing_data = json.load(f)
            except json.JSONDecodeError: print(f"Warning: Could not parse {log_file}")
        entry = {
            "expression": expression, "timestamp": int(time.time()), "alpha_id": alpha_data.get("id"),
            "fitness": alpha_data.get("is", {}).get("fitness"), "sharpe": alpha_data.get("is", {}).get("sharpe"),
            "turnover": alpha_data.get("is", {}).get("turnover"), "returns": alpha_data.get("is", {}).get("returns"),
            "grade": alpha_data.get("grade"), "checks": alpha_data.get("is", {}).get("checks")
        }
        existing_data.append(entry)
        try:
            with open(log_file, 'w') as f: json.dump(existing_data, f, indent=2)
            print(f"Logged promising WQ alpha to {log_file}")
        except IOError as e: print(f"Error saving hopeful alpha log: {e}")


    def get_results(self) -> List[Dict]:
        """Get all processed WQ simulation results."""
        # --- Keeping this function as is ---
        return self.results

    def fetch_submitted_alphas(self):
        """Fetch submitted alphas from the WorldQuant API with retry logic"""
        # --- Keeping this function as is ---
        url = "https://api.worldquantbrain.com/users/self/alphas"
        params = {"limit": 100, "offset": 0, "status!=": "UNSUBMITTED%1FIS-FAIL", "order": "-dateCreated", "hidden": "false"}
        max_retries, retry_delay = 3, 60
        for attempt in range(max_retries):
            try:
                response = self.sess.get(url, params=params, timeout=30)
                if response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', retry_delay))
                    logger.info(f"Rate limited fetching submitted alphas. Wait {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                return response.json().get("results", [])
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed fetch submitted alphas: {str(e)}. Retrying...")
                    time.sleep(retry_delay * (attempt + 1)) # Exponential backoff
                else:
                    logger.error(f"Failed fetch submitted alphas after {max_retries} attempts: {e}")
                    return []
        return []


# --- Helper functions (Keep As Is) ---
def extract_expressions(alphas):
    expressions = []
    if not alphas: return expressions
    for alpha in alphas:
        if alpha and isinstance(alpha, dict) and alpha.get("regular") and alpha["regular"].get("code"):
             perf_data = {}
             if alpha.get("is"): perf_data = {"sharpe": alpha["is"].get("sharpe"),"fitness": alpha["is"].get("fitness")}
             expressions.append({"expression": alpha["regular"]["code"], "performance": perf_data})
    return expressions

def is_similar_to_existing(new_expression, existing_expressions, similarity_threshold=0.75): # Slightly higher threshold
    if not existing_expressions: return False
    norm_new_expr = normalize_expression(new_expression)
    for existing in existing_expressions:
        if not isinstance(existing, dict) or "expression" not in existing: continue
        norm_existing_expr = normalize_expression(existing["expression"])
        if norm_new_expr == norm_existing_expr:
            logging.debug(f"Exact match found: {new_expression[:50]}...")
            return True
        if structural_similarity(norm_new_expr, norm_existing_expr) > similarity_threshold:
            logging.debug(f"Structurally similar ({structural_similarity(norm_new_expr, norm_existing_expr):.2f}): {new_expression[:50]}...")
            return True
    return False

def calculate_similarity(expr1: str, expr2: str) -> float:
    expr1_tokens = set(tokenize_expression(expr1))
    expr2_tokens = set(tokenize_expression(expr2))
    if not expr1_tokens or not expr2_tokens: return 0.0
    intersection = len(expr1_tokens.intersection(expr2_tokens))
    union = len(expr1_tokens.union(expr2_tokens))
    return intersection / union if union > 0 else 0.0

def structural_similarity(expr1, expr2):
    return calculate_similarity(expr1, expr2)

def normalize_expression(expr):
    expr = re.sub(r'\s+', '', expr.lower())
    return expr

def tokenize_expression(expr):
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+\.\d*|\.\d+|\d+|[(),+*/<>-]', expr) # Added hyphen
    return [token for token in tokens if token]


# --- Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description='Generate alpha factors using WorldQuant Brain API and OpenRouter')
    parser.add_argument('--credentials', type=str, default='./credential.txt',
                      help='Path to WQ credentials file (default: ./credential.txt)')
    parser.add_argument('--output-dir', type=str, default='./results_wq', # Changed default output dir
                      help='Directory to save WQ simulation results (default: ./results_wq)')
    parser.add_argument('--batch-size', type=int, default=5,
                      help='Number of alpha ideas to generate per AI batch (default: 5)')
    parser.add_argument('--sleep-time', type=int, default=60,
                      help='Sleep time between processing batches in seconds (default: 60)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')

    args = parser.parse_args()

    # Configure Logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_filename = 'alpha_generator_openrouter.log'
    log_file_handler = logging.FileHandler(log_filename)
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.basicConfig(level=logging.DEBUG, handlers=[log_file_handler, console_handler])
    logger.info(f"Logging level for console set to {args.log_level}")
    logger.info(f"Detailed logs available in {log_filename}")


    # Create output directory
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Results will be saved to {args.output_dir}")
    except OSError as e:
         logger.error(f"Failed to create output directory {args.output_dir}: {e}")
         return 1

    # Check for OpenRouter API Key
    if not os.getenv("OPENROUTER_API_KEY"):
       logging.error("FATAL: OPENROUTER_API_KEY environment variable not set. Please set it and restart.")
       return 1

    generator = None # Initialize generator to None
    total_successful_wq_sims = 0
    batch_number = 1
    try:
        generator = AlphaGenerator(args.credentials)

        print("Fetching initial WQ data fields and operators...")
        data_fields = generator.get_data_fields()
        operators = generator.get_operators()
        if not data_fields or not operators:
             logging.error("Failed to fetch initial WQ data fields or operators. Exiting.")
             return 1

        print(f"Starting continuous alpha mining (Batch Size: {args.batch_size}) using OpenRouter -> WQ")

        while True:
            try:
                logger.info(f"\n===== Processing Batch #{batch_number} =====")

                # 1. Generate Alpha Ideas via OpenRouter
                alpha_ideas = generator.generate_alpha_ideas(data_fields, operators)
                if not alpha_ideas:
                    logging.warning("No new alpha ideas generated by OpenRouter. Waiting...")
                    sleep(args.sleep_time * 2)
                    batch_number += 1
                    continue

                # 2. Filter against existing Alphas
                submitted_alphas_data = generator.fetch_submitted_alphas()
                existing_expressions = extract_expressions(submitted_alphas_data)
                logging.info(f"Fetched {len(existing_expressions)} existing WQ expressions for similarity check.")
                unique_ideas_to_test = [
                    idea for idea in alpha_ideas if not is_similar_to_existing(idea, existing_expressions)
                ]
                if len(unique_ideas_to_test) < len(alpha_ideas):
                     logging.info(f"Filtered out {len(alpha_ideas) - len(unique_ideas_to_test)} similar ideas.")
                if not unique_ideas_to_test:
                     logging.info("All generated ideas were similar to existing ones.")
                     sleep(args.sleep_time)
                     batch_number += 1
                     continue

                # 3. Test Unique Ideas on WorldQuant Brain
                logging.info(f"Submitting {len(unique_ideas_to_test)} unique alpha ideas to WQ Brain...")
                batch_successful_sims = generator.test_alpha_batch(unique_ideas_to_test)
                total_successful_wq_sims += batch_successful_sims

                # 4. Save accumulated results for the batch
                current_batch_results = generator.results
                if current_batch_results:
                     timestamp = int(time.time())
                     output_file = os.path.join(args.output_dir, f'batch_{batch_number}_wq_results_{timestamp}.json')
                     try:
                         with open(output_file, 'w') as f: json.dump(current_batch_results, f, indent=2)
                         logging.info(f"Batch {batch_number} WQ results saved to {output_file}")
                     except IOError as e: logger.error(f"Failed to save batch results: {e}")
                     generator.results = [] # Clear results after saving

                logging.info(f"Batch #{batch_number} complete. Successful WQ sims in batch: {batch_successful_sims}")
                logging.info(f"Total successful WQ sims so far: {total_successful_wq_sims}")
                batch_number += 1
                logging.info(f"Sleeping for {args.sleep_time} seconds...")
                sleep(args.sleep_time)

            except Exception as e:
                logging.error(f"Error processing batch #{batch_number}: {str(e)}", exc_info=True)
                logging.info("Sleeping for 5 minutes before next batch attempt...")
                sleep(300)
                batch_number += 1 # Ensure batch number increments even after error
                continue

    except KeyboardInterrupt:
        logger.info("\nCtrl+C received. Stopping alpha mining...")
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        return 1 # Indicate error
    finally:
        # Cleanup: Shutdown executor
        if generator and generator.executor:
             logging.info("Shutting down thread pool executor...")
             # Give threads a chance to finish current task, but don't wait indefinitely
             generator.executor.shutdown(wait=True, cancel_futures=False)
             logging.info("Executor shutdown complete.")
        logging.info(f"Total batches attempted: {batch_number - 1}")
        logging.info(f"Total successful WQ simulations found: {total_successful_wq_sims}")
        logging.info("Exiting.")
    return 0 # Indicate success

if __name__ == "__main__":
    exit(main())