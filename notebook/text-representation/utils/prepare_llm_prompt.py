## Create a XML template for LLM to generate synthentic

# Define your XML-like template
prompt_template = """You are tasked with generating realistic, detailed customer feedback based on the provided customer 
data profile. Analyze the entire customer profile below, paying attention to all sections (feedback category, complaints, 
engagement, offers, recency, customer info). Synthesize a specific, plausible customer feedback statement that reflects 
the original feedback category but elaborates on it, potentially incorporating details from other parts of the profile. 
Aim for a feedback statement a real customer might provide. Output ONLY the synthesized feedback statement, enclosed 
within <response></response> tags. Do not add any explanation or introductory text before or after the tags.

Customer Data Profile:
---
<original_feedback>
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
---

Synthesized Feedback:"""

# Apply the template row-wise
def generate_prompt_for_synthentic_data(row):
    return prompt_template.format(
        feedback=row.get('feedback', ''),
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
