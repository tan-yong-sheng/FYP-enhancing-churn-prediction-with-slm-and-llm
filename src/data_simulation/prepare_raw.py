'''
This script processes a CSV file containing customer data and generates individual text files 
for each record. It formats the data into a structured XML-like template and handles cases 
where specific fields are missing or have special values.
'''

import csv
import os

# Define input file and output directory
csv_file_path = 'data/input/churn.csv'
output_dir = 'tmp/synthetic_feedback/raw'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the template for the output text file
template = """<original_feedback>
  <feedback_category>{feedback}</feedback_category>
</original_feedback>

<complaints>
    <past_complaint>{past_complaint}</past_complaint>
    <complaint_status>{complaint_status}</complaint_status>
</complaints>

<engagement>
    <avg_time_spent>{avg_time_spent}</avg_time_spent>
    <avg_transaction_value>{avg_transaction_value}</avg_transaction_value>
    <avg_frequency_login_days>{avg_frequency_login_days}</avg_frequency_login_days>
</engagement>

<offers>
    <used_special_discount>{used_special_discount}</used_special_discount>
    <offer_application_preference>{offer_application_preference}</offer_application_preference>
    <preferred_offer_types>{preferred_offer_types}</preferred_offer_types>
</offers>

<recency>
    <last_visit_time>{last_visit_time}</last_visit_time>
    <days_since_last_login>{days_since_last_login}</days_since_last_login>
</recency>

<customer_info>
    <region_category>{region_category}</region_category>
    <membership_category>{membership_category}</membership_category>
</customer_info>
"""

try:
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        print(f"Reading from {csv_file_path}...")
        count = 0
        for row in reader:
            try:
                feedback_value = row.get('feedback', '')

                # Check if feedback is 'No reason specified'
                if feedback_value == 'No reason specified':
                    formatted_content = '<response>No reason specified</response>'
                else:
                    # Format the template with data from the current row
                    # Handle potential missing keys gracefully, defaulting to empty string
                    formatted_content = template.format(
                        feedback=feedback_value,
                        past_complaint=row.get('past_complaint', ''),
                    complaint_status=row.get('complaint_status', ''),
                    avg_time_spent=row.get('avg_time_spent', ''),
                    avg_transaction_value=row.get('avg_transaction_value', ''),
                    avg_frequency_login_days=row.get('avg_frequency_login_days', ''),
                    used_special_discount=row.get('used_special_discount', ''),
                    offer_application_preference=row.get('offer_application_preference', ''),
                    preferred_offer_types=row.get('preferred_offer_types', ''),
                    last_visit_time=row.get('last_visit_time', ''),
                    days_since_last_login=row.get('days_since_last_login', ''),
                    region_category=row.get('region_category', ''),
                    membership_category=row.get('membership_category', '')
                )

                # Create the output filename using the 'index' column
                output_filename = os.path.join(output_dir, f"{row['index']}.txt")

                # Write the formatted content to the output file
                with open(output_filename, mode='w', encoding='utf-8') as outfile:
                    outfile.write(formatted_content)

                count += 1
                if count % 1000 == 0: # Print progress every 1000 files
                    print(f"Processed {count} records...")

            except KeyError as e:
                print(f"Skipping row due to missing key: {e} in row with index {row.get('index', 'N/A')}")
            except Exception as e:
                print(f"An error occurred processing row with index {row.get('index', 'N/A')}: {e}")

    print(f"\nSuccessfully processed {count} records.")
    print(f"Text files created in the '{output_dir}' directory.")

except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
