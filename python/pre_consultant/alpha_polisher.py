# File: python/pre_consultant/alpha_polisher.py
# Modified to use OpenRouter API instead of MoonshotAI or direct Google Gemini

import argparse
import requests # Make sure requests is installed
import json
import os
import re
from time import sleep
from requests.auth import HTTPBasicAuth # For WQ Auth
from typing import List, Dict, Optional
import time
import logging

# Configure logger
logger = logging.getLogger(__name__)
# Ensure logging is configured in the main block

# --- AlphaPolisher Class ---
class AlphaPolisher:
    def __init__(self, credentials_path: str):
        logger.info("Initializing AlphaPolisher...")
        self.sess = requests.Session() # Session for WQ Brain calls
        self.credentials_path = credentials_path
        self.setup_auth(credentials_path)
        self.operators = self.fetch_operators()

        # Check for OpenRouter API Key
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            logger.error("FATAL: OPENROUTER_API_KEY environment variable not set.")
            raise ValueError("OPENROUTER_API_KEY environment variable is required.")
        else:
            logger.info("OpenRouter API Key found.")

        # Optional: Set Referer and Title for OpenRouter attribution
        self.openrouter_site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
        self.openrouter_site_name = os.getenv("OPENROUTER_SITE_NAME", "WQAlphaPolisher")

        logger.info("AlphaPolisher initialized successfully using OpenRouter.")

    def setup_auth(self, credentials_path: str) -> None:
        """Set up authentication with WorldQuant Brain."""
        logger.info(f"Loading WQ credentials from {credentials_path}")
        try:
            with open(credentials_path) as f:
                credentials = json.load(f)
            if not isinstance(credentials, list) or len(credentials) != 2:
                raise ValueError("Invalid WQ credentials format in file.")
            username, password = credentials
            self.sess.auth = HTTPBasicAuth(username, password)
            logger.info("Authenticating with WorldQuant Brain...")
            response = self.sess.post('https://api.worldquantbrain.com/authentication', timeout=15)
            logger.debug(f"WQ Authentication response status: {response.status_code}")
            response.raise_for_status()
            logger.info("WorldQuant Brain authentication successful")
        except FileNotFoundError:
            logger.error(f"Credentials file not found: {credentials_path}")
            raise
        except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:
            logger.error(f"WorldQuant Brain authentication failed: {str(e)}")
            raise Exception(f"WQ Authentication failed: {str(e)}") from e

    def fetch_operators(self) -> Optional[List[Dict]]:
        """Fetch available operators from WorldQuant Brain API."""
        logger.info("Fetching available WorldQuant operators...")
        try:
            response = self.sess.get('https://api.worldquantbrain.com/operators', timeout=20)
            logger.debug(f"WQ Operators response status: {response.status_code}")
            response.raise_for_status()
            operators_data = response.json()
            if isinstance(operators_data, list):
                 logger.info(f"Successfully fetched {len(operators_data)} WQ operators")
                 return operators_data
            else:
                 logger.error(f"Unexpected format for WQ operators response: Type is {type(operators_data)}")
                 return None
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Error fetching WQ operators: {str(e)}")
            return None

    # --- MODIFIED Function to use OpenRouter ---
    def analyze_alpha(self, expression: str) -> Dict:
        """Analyze the alpha expression and provide insights using OpenRouter API."""
        logger.info(f"Analyzing expression with OpenRouter: {expression[:100]}...")

        # Prepare operators info for the prompt
        operators_prompt_info = "WorldQuant operator information not available."
        if self.operators:
             try:
                  # Limit prompt size
                  limited_operators = self.operators[:100]
                  operators_prompt_info = f"Available WorldQuant operators (sample):\n{json.dumps(limited_operators, indent=2)}"
                  if len(operators_prompt_info) > 15000: # Further limit if huge
                       operators_prompt_info = f"Available WorldQuant operators (sample):\n{json.dumps(limited_operators[:50], indent=2)}"
                  logger.debug(f"Using operators info (truncated): {operators_prompt_info[:500]}...")
             except Exception as json_err:
                  logger.warning(f"Could not serialize operators for prompt: {json_err}")
                  operators_prompt_info = "Could not load WorldQuant operator information."
        else:
            logger.warning("WQ Operator data not fetched, proceeding without it in the prompt.")

        # Construct the prompt for OpenRouter (similar to Gemini version)
        prompt = f"""You are an expert quantitative analyst specializing in WorldQuant Brain alpha expressions (FASTEXPR language).
{operators_prompt_info}

Please analyze this WorldQuant alpha expression:

{expression}


Provide a concise analysis covering these points:
1.  **Strategy/Inefficiency:** What market pattern, anomaly, or strategy does this alpha likely try to capture?
2.  **Key Components:** Break down the expression. What is the role of each main operator and data field?
3.  **Potential Strengths:** What might make this alpha perform well under certain conditions?
4.  **Potential Risks/Limitations:** What are the potential downsides, risks, or scenarios where it might fail? (e.g., high turnover, sensitivity to market regime, data sparsity).
5.  **Validity Check:** Based on the sample operators provided, are the operators and their usage plausible in the FASTEXPR language? (Identify any obvious syntax errors or type mismatches if possible).
6.  **Improvement Suggestion:** Offer one specific, actionable suggestion for potentially improving the alpha (e.g., different parameter, adding normalization, combining with another factor).

Keep the analysis clear and focused on these points.
"""
        # --- OpenRouter API Call ---
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.openrouter_site_url,
            "X-Title": self.openrouter_site_name,
        }
        # Choose a suitable model on OpenRouter for analysis
        openrouter_model = "google/gemini-pro" # Or another capable model like "anthropic/claude-3-haiku"

        data = json.dumps({
            "model": openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            # "max_tokens": 1500, # Optional
        })

        analysis_text = f"Error: Analysis failed to generate." # Default error message
        try:
            logger.info(f"Sending analysis request to OpenRouter (Model: {openrouter_model})...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=data,
                timeout=90 # Increased timeout
            )
            logger.debug(f"OpenRouter analysis response status: {response.status_code}")
            response.raise_for_status()
            response_data = response.json()

            if response_data.get("choices"):
                analysis_text = response_data["choices"][0]["message"]["content"]
                logger.info("Successfully generated analysis via OpenRouter.")
            else:
                 error_detail = response_data.get("error", {}).get("message", "No choices found in response.")
                 analysis_text = f"Error: OpenRouter analysis response missing 'choices'. Detail: {error_detail}"
                 logger.error(analysis_text)

        except requests.exceptions.Timeout:
             error_msg = "Timeout connecting to OpenRouter API for analysis."
             logger.error(error_msg)
             analysis_text = f"Error: {error_msg}"
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else "N/A"
            response_text = e.response.text[:500] if e.response is not None else "N/A"
            error_msg = f"OpenRouter analysis request failed (Status: {status_code}): {e}. Response: {response_text}"
            logger.error(error_msg)
            if status_code == 401: error_msg += " Check API Key."
            elif status_code == 402: error_msg += " Check Credits."
            elif status_code == 429: error_msg += " Rate Limited."
            analysis_text = f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error during OpenRouter analysis API call: {str(e)}"
            logger.error(error_msg, exc_info=True)
            analysis_text = f"Error: {error_msg}"

        return {"analysis": analysis_text}

    # --- MODIFIED Function to use OpenRouter ---
    def polish_expression(self, expression: str, user_requirements: str = "") -> Dict:
        """Request polished version of the alpha expression from OpenRouter API."""
        logger.info(f"Polishing expression with OpenRouter: {expression[:100]}...")
        if user_requirements:
            logger.info(f"User requirements: {user_requirements}")

        # Prepare operators info (similar to analyze_alpha)
        operators_prompt_info = "WorldQuant operator information not available."
        if self.operators:
             try:
                  limited_operators = self.operators[:100]
                  operators_prompt_info = f"Available WorldQuant operators (sample):\n{json.dumps(limited_operators, indent=2)}"
                  if len(operators_prompt_info) > 15000:
                       operators_prompt_info = f"Available WorldQuant operators (sample):\n{json.dumps(limited_operators[:50], indent=2)}"
                  logger.debug(f"Using operators info (truncated): {operators_prompt_info[:500]}...")
             except Exception as json_err:
                  logger.warning(f"Could not serialize operators for prompt: {json_err}")
                  operators_prompt_info = "Could not load WorldQuant operator information."
        else:
             logger.warning("WQ Operator data not fetched, proceeding without it in the prompt.")


        # Construct the prompt for OpenRouter (similar to Gemini version)
        user_message_parts = [
            f"You are an expert quantitative analyst specializing in improving WorldQuant Brain alpha expressions (FASTEXPR language).",
            f"{operators_prompt_info}\n",
            f"Please carefully polish the following WorldQuant alpha expression to potentially improve its performance (e.g., Sharpe ratio, fitness, IR) while trying to maintain its core strategic idea.",
            f"Original Expression:\n```\n{expression}\n```"
        ]
        if user_requirements:
            user_message_parts.append(f"\nConsider these specific requirements:\n{user_requirements}")
        user_message_parts.append("\nMake thoughtful changes, such as adjusting parameters, adding normalization (rank, zscore), applying smoothing (ts_mean), handling outliers (winsorize), or combining with relevant complementary factors (like volume or volatility).")
        user_message_parts.append("\nReturn ONLY the single, complete, polished FASTEXPR expression. Do not include explanations, comments, backticks, markdown formatting, or any other text.")
        user_message_parts.append("Example of desired output format: -ts_corr(rank(close), rank(volume), 20)")
        prompt = "\n".join(user_message_parts)

        # --- OpenRouter API Call ---
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.openrouter_site_url,
            "X-Title": self.openrouter_site_name,
        }
        # Choose a model suitable for code modification
        openrouter_model = "google/gemini-pro" # Or "anthropic/claude-3-haiku", "mistralai/mixtral-8x7b-instruct" etc.

        data = json.dumps({
            "model": openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4, # Lower temp for focused polishing
            # "max_tokens": 512, # Optional
        })

        polished_expr = "Error: Polishing failed to generate." # Default error
        try:
            logger.info(f"Sending polish request to OpenRouter (Model: {openrouter_model})...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=data,
                timeout=90 # Increased timeout
            )
            logger.debug(f"OpenRouter polish response status: {response.status_code}")
            response.raise_for_status()
            response_data = response.json()

            if response_data.get("choices"):
                raw_polished = response_data["choices"][0]["message"]["content"].strip()
                logger.debug(f"Raw OpenRouter polish response: {raw_polished}")

                # --- Clean the response ---
                # Remove potential markdown code blocks
                cleaned = re.sub(r'^```(fast)?expr\s*', '', raw_polished, flags=re.IGNORECASE | re.MULTILINE)
                cleaned = re.sub(r'```$', '', cleaned).strip()
                # Attempt to extract the last line if multiple lines were returned erroneously
                lines = cleaned.split('\n')
                if len(lines) > 1 and '(' in lines[-1] and ')' in lines[-1]:
                     final_expr = lines[-1].strip()
                     logger.warning("OpenRouter polish returned multiple lines; extracted last line.")
                else:
                     final_expr = cleaned # Assume single line is correct

                # Final validation check
                if '(' not in final_expr or ')' not in final_expr or len(final_expr.split()) < 2 : # Basic check
                     logger.error(f"OpenRouter polish result validation failed. Result: '{final_expr}'. Raw: '{raw_polished}'")
                     polished_expr = f"Error: Polished result validation failed. Raw response: {raw_polished}"
                else:
                     polished_expr = final_expr
                     logger.info("Successfully polished expression via OpenRouter.")
                     logger.debug(f"Cleaned Polished result: {polished_expr}")
            else:
                 error_detail = response_data.get("error", {}).get("message", "No choices found in response.")
                 polished_expr = f"Error: OpenRouter polish response missing 'choices'. Detail: {error_detail}"
                 logger.error(polished_expr)

        except requests.exceptions.Timeout:
             error_msg = "Timeout connecting to OpenRouter API for polishing."
             logger.error(error_msg)
             polished_expr = f"Error: {error_msg}"
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else "N/A"
            response_text = e.response.text[:500] if e.response is not None else "N/A"
            error_msg = f"OpenRouter polish request failed (Status: {status_code}): {e}. Response: {response_text}"
            logger.error(error_msg)
            if status_code == 401: error_msg += " Check API Key."
            elif status_code == 402: error_msg += " Check Credits."
            elif status_code == 429: error_msg += " Rate Limited."
            polished_expr = f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error during OpenRouter polish API call: {str(e)}"
            logger.error(error_msg, exc_info=True)
            polished_expr = f"Error: {error_msg}"

        return {"polished_expression": polished_expr}


    def simulate_alpha(self, expression: str) -> Dict:
        """Simulate the alpha expression using WorldQuant Brain API."""
        # --- Keeping this function as is (interacts only with WQ Brain) ---
        logger.info(f"Simulating WQ expression: {expression[:100]}...")
        url = 'https://api.worldquantbrain.com/simulations'
        data = {
            'type': 'REGULAR',
            'settings': {'instrumentType': 'EQUITY','region': 'USA','universe': 'TOP3000','delay': 1,'decay': 0,'neutralization': 'INDUSTRY','truncation': 0.08,'pasteurization': 'ON','unitHandling': 'VERIFY','nanHandling': 'OFF','language': 'FASTEXPR','visualization': False,},
            'regular': expression
        }
        logger.debug(f"WQ Simulation request data: {json.dumps(data, indent=2)}")
        sim_resp = None
        try:
            logger.info("Sending WQ simulation request...")
            sim_resp = self.sess.post(url, json=data, timeout=30)
            logger.debug(f"WQ Simulation creation response status: {sim_resp.status_code}")
            if sim_resp.status_code == 401:
                logger.warning("WQ Auth expired on simulation, refreshing...")
                self.setup_auth(self.credentials_path)
                sim_resp = self.sess.post(url, json=data, timeout=30)
                logger.debug(f"Retry WQ Simulation creation status: {sim_resp.status_code}")
            if sim_resp.status_code == 400 and "invalid expression" in sim_resp.text.lower():
                 error_msg = f"WQ rejected invalid expression: {expression}. Error: {sim_resp.text}"
                 logger.error(error_msg)
                 return {"status": "error", "message": error_msg}
            if sim_resp.status_code == 429:
                 error_msg = f"WQ Rate limit hit simulation submission."
                 logger.error(error_msg)
                 return {"status": "error", "message": error_msg}
            sim_resp.raise_for_status() # Raise for other bad statuses
            if sim_resp.status_code == 201:
                progress_url = sim_resp.headers.get('location')
                if not progress_url:
                     error_msg = "WQ Sim created (201) but no progress URL."
                     logger.error(error_msg)
                     return {"status": "error", "message": error_msg}
                logger.info(f"WQ Sim submitted. Monitoring: {progress_url}")
                monitor_start_time = time.time()
                MONITOR_TIMEOUT = 60 * 10
                while time.time() - monitor_start_time < MONITOR_TIMEOUT:
                    sleep(5)
                    try:
                        result_resp = self.sess.get(progress_url, timeout=20)
                        if result_resp.status_code == 429:
                             retry_after = int(result_resp.headers.get('Retry-After', 5))
                             logger.warning(f"WQ rate limit monitoring. Wait {retry_after}s...")
                             sleep(retry_after)
                             continue
                        if result_resp.status_code == 401:
                             logger.warning("WQ Auth expired monitoring. Re-authenticating...")
                             self.setup_auth(self.credentials_path)
                             continue
                        result_resp.raise_for_status()
                        if not result_resp.text.strip():
                             logger.debug(f"WQ Sim {progress_url} processing...")
                             continue
                        try: sim_data = result_resp.json()
                        except json.JSONDecodeError:
                              logger.warning(f"WQ Sim {progress_url} not JSON yet: {result_resp.text[:100]}...")
                              continue
                        status = sim_data.get('status')
                        sim_id = sim_data.get('id', 'N/A')
                        logger.info(f"WQ Sim {sim_id} status: {status}")
                        if status == 'COMPLETE':
                            logger.info("WQ Sim complete.")
                            alpha_id = sim_data.get("alpha")
                            if alpha_id:
                                 logger.info(f"Fetching details for alpha {alpha_id}...")
                                 details_resp = self.sess.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}", timeout=20)
                                 if details_resp.status_code == 200:
                                      alpha_details = details_resp.json()
                                      logger.info("WQ alpha details retrieved.")
                                      return {"status": "success", "simulation_summary": sim_data, "alpha_details": alpha_details}
                                 else:
                                      error_msg = f"Sim complete, but failed fetch alpha details {alpha_id}: {details_resp.status_code}"
                                      logger.error(error_msg)
                                      return {"status": "error", "message": error_msg, "simulation_summary": sim_data}
                            else:
                                 logger.warning(f"Sim {sim_id} complete but no alpha ID.")
                                 return {"status": "success", "simulation_summary": sim_data, "alpha_details": None}
                        elif status in ['FAILED', 'ERROR']:
                            error_msg = f"WQ Sim failed: {sim_data.get('message', 'Unknown error')}"
                            logger.error(error_msg)
                            return {"status": "error", "message": error_msg, "simulation_summary": sim_data}
                        elif status == 'RUNNING': logger.debug(f"WQ Sim {sim_id} running...")
                        else: logger.warning(f"Unknown WQ sim status: {status} for {sim_id}")
                    except requests.exceptions.Timeout: logger.warning(f"Timeout check WQ sim status {progress_url}. Retrying.")
                    except requests.exceptions.RequestException as monitor_err: logger.warning(f"Network error monitoring WQ sim {progress_url}: {monitor_err}. Retrying.")
                error_msg = f"WQ Sim monitoring timed out: {progress_url}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            else: # Initial submission failed (not 201)
                 error_msg = f"Failed create WQ sim. Status: {sim_resp.status_code}. Response: {sim_resp.text[:500]}"
                 logger.error(error_msg)
                 return {"status": "error", "message": error_msg}
        except requests.exceptions.Timeout:
             error_msg = f"Timeout initial WQ sim request: {expression[:100]}..."
             logger.error(error_msg)
             return {"status": "error", "message": error_msg}
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error WQ sim: {str(e)}"
            logger.error(error_msg)
            if sim_resp is not None: logger.error(f"Last WQ Response: {sim_resp.status_code} {sim_resp.text[:200]}")
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error WQ sim: {str(e)}"
            logger.exception("Traceback unexpected WQ sim error:")
            return {"status": "error", "message": error_msg}


# --- Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description='Polish and analyze WorldQuant alpha expressions using OpenRouter API')
    parser.add_argument('--credentials', type=str, required=True, help='Path to WorldQuant credentials file')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')

    args = parser.parse_args()

    # Configure Logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_filename = 'alpha_polisher_openrouter.log' # Specific log file
    log_file_handler = logging.FileHandler(log_filename)
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.basicConfig(level=logging.DEBUG, handlers=[log_file_handler, console_handler])
    logger.info(f"Logging level for console set to {args.log_level}")
    logger.info(f"Detailed logs available in {log_filename}")

    # Check for OpenRouter API Key
    if not os.getenv("OPENROUTER_API_KEY"):
       logger.error("FATAL: OPENROUTER_API_KEY environment variable not set. Please set it and restart.")
       return 1

    polisher = None # Initialize to None
    try:
        logger.info("Starting Alpha Polisher using OpenRouter...")
        polisher = AlphaPolisher(args.credentials) # Instantiate polisher

        while True:
            try:
                print("\n" + "="*50)
                print("Enter WorldQuant alpha expression (or type 'quit' to exit):")
                expression = input("Expression: ").strip()
                if expression.lower() == 'quit':
                    logger.info("User requested quit.")
                    break
                if not expression:
                     print("Expression empty. Try again.")
                     continue

                print("\nEnter polishing requirements (optional, press Enter):")
                print("Examples: Improve IR, Reduce turnover < 0.5, Add volume confirmation")
                user_requirements = input("Requirements: ").strip()

                logger.info(f"Processing expression: {expression}")
                if user_requirements: logger.info(f"User requirements: {user_requirements}")

                # 1. Analyze Original (OpenRouter)
                print("\nAnalyzing original (via OpenRouter)...")
                analysis_result = polisher.analyze_alpha(expression)
                print("\n--- Analysis ---")
                print(analysis_result.get("analysis", "Analysis failed."))
                print("--- End Analysis ---")

                # 2. Polish Expression (OpenRouter)
                print("\nPolishing expression (via OpenRouter)...")
                polish_result = polisher.polish_expression(expression, user_requirements)
                polished_expression = polish_result.get("polished_expression")

                if polished_expression and "Error:" not in polished_expression:
                     print(f"\nPolished expression:\n{polished_expression}")

                     # 3. Simulate Original (WQ Brain)
                     print("\nSimulating ORIGINAL on WQ Brain...")
                     sim_result_orig = polisher.simulate_alpha(expression)
                     print("\n--- Original WQ Simulation Result ---")
                     if sim_result_orig.get("status") == "success":
                          details = sim_result_orig.get("alpha_details", {}).get("is", {})
                          print(f"  Status: Success")
                          print(f"  Sharpe: {details.get('sharpe', 'N/A'):.3f}")
                          print(f"  Fitness: {details.get('fitness', 'N/A'):.3f}")
                          print(f"  Turnover: {details.get('turnover', 'N/A'):.3f}")
                     else:
                          print(f"  Status: Error / Message: {sim_result_orig.get('message', 'N/A')}")
                     print("--- End Original WQ Result ---")

                     # 4. Simulate Polished (WQ Brain)
                     print("\nSimulating POLISHED on WQ Brain...")
                     sim_result_polished = polisher.simulate_alpha(polished_expression)
                     print("\n--- Polished WQ Simulation Result ---")
                     if sim_result_polished.get("status") == "success":
                           details = sim_result_polished.get("alpha_details", {}).get("is", {})
                           print(f"  Status: Success")
                           print(f"  Sharpe: {details.get('sharpe', 'N/A'):.3f}")
                           print(f"  Fitness: {details.get('fitness', 'N/A'):.3f}")
                           print(f"  Turnover: {details.get('turnover', 'N/A'):.3f}")
                     else:
                           print(f"  Status: Error / Message: {sim_result_polished.get('message', 'N/A')}")
                     print("--- End Polished WQ Result ---")
                else:
                    print("\nPolishing failed or returned an error:")
                    print(polished_expression) # Show polishing error

            except EOFError: logger.warning("EOF received, exiting loop."); break
            except Exception as loop_err:
                 logger.error(f"Error during processing loop: {loop_err}", exc_info=True)
                 print(f"An error occurred: {loop_err}. Please try again or type 'quit'.")

        logger.info("Alpha Polisher finished.")
    except KeyboardInterrupt:
        logger.info("\nCtrl+C received. Exiting Alpha Polisher.")
    except Exception as e:
        logger.error(f"Fatal error during Alpha Polisher execution: {str(e)}", exc_info=True)
        return 1
    finally:
         # Any cleanup if needed
         logger.info("Exiting Alpha Polisher.")
    return 0

if __name__ == "__main__":
    exit(main())