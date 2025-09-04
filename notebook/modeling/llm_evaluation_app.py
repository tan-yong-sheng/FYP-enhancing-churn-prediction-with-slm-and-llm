import gradio as gr
import pandas as pd
import json
import os
from typing import Dict, Any

class LLMEvaluationApp:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.current_index = 0
        self.load_data()
    
    def load_data(self):
        """Load the CSV data and filter out empty rows"""
        try:
            self.df = pd.read_csv(self.csv_path, compression='gzip')
            # Filter out rows where essential columns are empty
            self.df = self.df.dropna(subset=['explanation', 'meta_data_description'])
            self.df = self.df.reset_index(drop=True)
            print(f"Loaded {len(self.df)} valid rows for evaluation")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def get_current_row_data(self):
        """Get data for current row"""
        if self.df.empty or self.current_index >= len(self.df):
            return None
        
        row = self.df.iloc[self.current_index]
        return {
            'meta_data_description': row.get('meta_data_description', 'N/A'),
            'explanation': row.get('explanation', 'N/A'),
            'shap_values': str(row.get('shap_values', 'N/A')),
            'top_5_shap_values': str(row.get('top_5_shap_values', 'N/A')),
            'predicted_label': row.get('predicted_label', 'N/A'),
            'original_index': row.get('original_index', self.current_index + 1),
            'finished': row.get('finished', False),
            'human_evaluation': row.get('human_evaluation', None)
        }
    
    def get_progress_info(self):
        """Get progress information"""
        if self.df.empty:
            return "No data loaded"
        
        completed_count = self.df['finished'].sum()
        total_count = len(self.df)
        progress_percent = (completed_count / total_count * 100) if total_count > 0 else 0
        
        status = "COMPLETED" if self.df.iloc[self.current_index]['finished'] else "PENDING"
        
        info = f"Row {self.current_index + 1} of {total_count} (Original Index: {self.df.iloc[self.current_index].get('original_index', self.current_index + 1)}) - Status: {status}"
        progress = f"Progress: {completed_count}/{total_count} completed ({progress_percent:.1f}%)"
        
        return f"{info}\n{progress}"
    
    def save_evaluation(self, factual_score: int, factual_justification: str,
                       completeness_score: int, completeness_justification: str,
                       clarity_score: int, clarity_justification: str):
        """Save evaluation for current row"""
        if self.df.empty or self.current_index >= len(self.df):
            return "Error: No data to evaluate"
        
        # Calculate overall average
        overall_average = (factual_score + completeness_score + clarity_score) / 3
        
        # Create evaluation object
        evaluation = {
            'factual_accuracy': {
                'score': factual_score,
                'justification': factual_justification
            },
            'completeness_integration': {
                'score': completeness_score,
                'justification': completeness_justification
            },
            'clarity_readability': {
                'score': clarity_score,
                'justification': clarity_justification
            },
            'overall_average': round(overall_average, 2)
        }
        
        # Save to dataframe
        self.df.at[self.current_index, 'human_evaluation'] = json.dumps(evaluation)
        self.df.at[self.current_index, 'finished'] = True
        
        # Save to file
        self.save_to_file()
        
        return f"Evaluation saved successfully! Overall Average: {overall_average:.2f}"
    
    def save_to_file(self):
        """Save dataframe back to CSV file"""
        try:
            self.df.to_csv(self.csv_path, compression='gzip', index=False)
            print("Data saved to file")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    def next_row(self):
        """Move to next row"""
        if self.current_index < len(self.df) - 1:
            self.current_index += 1
        return self.get_current_row_data()
    
    def previous_row(self):
        """Move to previous row"""
        if self.current_index > 0:
            self.current_index -= 1
        return self.get_current_row_data()
    
    def skip_to_next_unfinished(self):
        """Skip to next unfinished row"""
        start_index = self.current_index + 1
        for i in range(start_index, len(self.df)):
            if not self.df.iloc[i]['finished']:
                self.current_index = i
                return self.get_current_row_data()
        
        # If no unfinished found, stay at current
        return self.get_current_row_data()

    def go_to_row(self, row_number: int):
        """Go to specific row number (1-indexed)"""
        if 1 <= row_number <= len(self.df):
            self.current_index = row_number - 1  # Convert to 0-indexed
            return self.get_current_row_data()
        else:
            # Invalid row number, stay at current position
            return self.get_current_row_data()

# Initialize the app - Try multiple possible paths
import os
possible_paths = [
    "./data/output/evaluation_data_shuffled.csv.gz",
    "../../data/output/evaluation_data_shuffled.csv.gz",
    "./notebook/modeling/../../data/output/evaluation_data_shuffled.csv.gz"
]

CSV_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        CSV_PATH = path
        print(f"Found data file at: {path}")
        break

if CSV_PATH is None:
    print("Error: Could not find evaluation_data_shuffled.csv.gz")
    print("Please run this script from the project root directory or from notebook/modeling/")
    CSV_PATH = "./data/output/evaluation_data_shuffled.csv.gz"  # Default fallback

app = LLMEvaluationApp(CSV_PATH)

def count_words(text):
    """Count words in text"""
    if not text or text == 'N/A':
        return 0
    return len(text.split())

def update_display():
    """Update the display with current row data"""
    data = app.get_current_row_data()
    progress_info = app.get_progress_info()
    
    if data is None:
        return "No data available", "", "", "", "", "", progress_info, "", None, "", None, "", None, ""
    
    # Load existing evaluation if available
    existing_eval = ""
    factual_score, factual_just = None, ""
    completeness_score, completeness_just = None, ""
    clarity_score, clarity_just = None, ""
    
    if data['human_evaluation']:
        try:
            eval_data = json.loads(data['human_evaluation'])
            existing_eval = json.dumps(eval_data, indent=2)
            
            factual_score = eval_data.get('factual_accuracy', {}).get('score', None)
            factual_just = eval_data.get('factual_accuracy', {}).get('justification', "")
            completeness_score = eval_data.get('completeness_integration', {}).get('score', None)
            completeness_just = eval_data.get('completeness_integration', {}).get('justification', "")
            clarity_score = eval_data.get('clarity_readability', {}).get('score', None)
            clarity_just = eval_data.get('clarity_readability', {}).get('justification', "")
        except:
            pass
    
    # Count words in explanation
    explanation_text = data['explanation']
    word_count = count_words(explanation_text)
    word_count_display = f"Word Count: {word_count} words"
    
    return (
        data['predicted_label'],
        data['top_5_shap_values'],
        word_count_display,
        data['explanation'],
        data['shap_values'],
        data['meta_data_description'],
        progress_info,
        existing_eval,
        factual_score,
        factual_just,
        completeness_score,
        completeness_just,
        clarity_score,
        clarity_just
    )

def submit_evaluation(factual_score, factual_justification, completeness_score, 
                     completeness_justification, clarity_score, clarity_justification):
    """Submit evaluation and move to next unfinished row"""
    # Check if all required fields are filled
    if (factual_score is None or not factual_justification.strip() or 
        completeness_score is None or not completeness_justification.strip() or
        clarity_score is None or not clarity_justification.strip()):
        # Return error message and keep current display
        current_data = update_display()
        return ("Please fill in all fields",) + current_data
    
    result = app.save_evaluation(int(factual_score), factual_justification,
                                int(completeness_score), completeness_justification,
                                int(clarity_score), clarity_justification)
    
    # Move to next unfinished row
    app.skip_to_next_unfinished()
    
    # Get updated display data and prepend result message
    updated_data = update_display()
    return (result,) + updated_data

def move_next():
    app.next_row()
    return update_display()

def move_previous():
    app.previous_row()
    return update_display()

def skip_unfinished():
    app.skip_to_next_unfinished()
    return update_display()

def go_to_row(row_number):
    """Go to specific row number"""
    if row_number is None or row_number < 1 or row_number > len(app.df):
        # Invalid input, return current display with error message
        current_data = update_display()
        error_msg = f"Invalid row number. Please enter a number between 1 and {len(app.df)}"
        return (error_msg,) + current_data
    
    app.go_to_row(int(row_number))
    updated_data = update_display()
    success_msg = f"Navigated to row {row_number}"
    return (success_msg,) + updated_data

# Create Gradio interface
with gr.Blocks(title="LLM Narrative Evaluation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LLM Narrative Evaluation Interface")
    
    with gr.Row():
        with gr.Column():
            progress_text = gr.Textbox(label="Progress", interactive=False, lines=2)
    
    with gr.Row():
        prev_btn = gr.Button("← Previous", variant="secondary")
        next_btn = gr.Button("Next →", variant="secondary") 
        skip_btn = gr.Button("Skip to Next Unfinished", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=2):
            row_number_input = gr.Number(label="Go to Row Number", minimum=1, precision=0, value=1)
        with gr.Column(scale=1):
            go_to_btn = gr.Button("Go to Row", variant="secondary")
        with gr.Column(scale=3):
            navigation_result = gr.Textbox(label="Navigation Result", interactive=False)
    
    with gr.Tab("Data to Evaluate"):
        predicted_label = gr.Textbox(label="Predicted Label", interactive=False)
        top5_shap = gr.Textbox(label="Top 5 SHAP Values", interactive=False)
        word_counter = gr.Textbox(label="Explanation Statistics", interactive=False)
        explanation = gr.Textbox(label="🔍 LLM-Generated Explanation (TO EVALUATE)", 
                                lines=5, interactive=False)
        shap_values = gr.Textbox(label="SHAP Values (Full)", lines=4, interactive=False)
        meta_desc = gr.Textbox(label="Meta Data Description", lines=3, interactive=False)
    
    with gr.Tab("Evaluation Form"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Factual Accuracy (1-5)")
                factual_score = gr.Number(label="Score", minimum=1, maximum=5, precision=0, value=None)
                factual_justification = gr.Textbox(label="Justification", lines=3, 
                                                 placeholder="Explain your factual accuracy rating...")
            
            with gr.Column():
                gr.Markdown("### Completeness Integration (1-5)")
                completeness_score = gr.Number(label="Score", minimum=1, maximum=5, precision=0, value=None)
                completeness_justification = gr.Textbox(label="Justification", lines=3,
                                                       placeholder="Explain your completeness rating...")
            
            with gr.Column():
                gr.Markdown("### Clarity Readability (1-5)")
                clarity_score = gr.Number(label="Score", minimum=1, maximum=5, precision=0, value=None)
                clarity_justification = gr.Textbox(label="Justification", lines=3,
                                                 placeholder="Explain your clarity rating...")
        
        submit_btn = gr.Button("Submit Evaluation & Move to Next", variant="primary", size="lg")
        result_text = gr.Textbox(label="Result", interactive=False)
    
    with gr.Tab("Existing Evaluation"):
        existing_eval = gr.Textbox(label="Previous Evaluation (if any)", 
                                  lines=10, interactive=False)
    
    # Event handlers
    submit_btn.click(
        submit_evaluation,
        inputs=[factual_score, factual_justification, completeness_score, 
                completeness_justification, clarity_score, clarity_justification],
        outputs=[result_text, predicted_label, top5_shap, word_counter, explanation, shap_values, meta_desc, 
                progress_text, existing_eval,
                factual_score, factual_justification, completeness_score, 
                completeness_justification, clarity_score, clarity_justification]
    )
    
    next_btn.click(move_next, outputs=[predicted_label, top5_shap, word_counter, explanation, shap_values, meta_desc, 
                                      progress_text, existing_eval,
                                      factual_score, factual_justification, completeness_score, 
                                      completeness_justification, clarity_score, clarity_justification])
    
    prev_btn.click(move_previous, outputs=[predicted_label, top5_shap, word_counter, explanation, shap_values, meta_desc, 
                                          progress_text, existing_eval,
                                          factual_score, factual_justification, completeness_score, 
                                          completeness_justification, clarity_score, clarity_justification])
    
    skip_btn.click(skip_unfinished, outputs=[predicted_label, top5_shap, word_counter, explanation, shap_values, meta_desc, 
                                            progress_text, existing_eval,
                                            factual_score, factual_justification, completeness_score, 
                                            completeness_justification, clarity_score, clarity_justification])
    
    go_to_btn.click(go_to_row, 
                   inputs=[row_number_input],
                   outputs=[navigation_result, predicted_label, top5_shap, word_counter, explanation, shap_values, meta_desc, 
                           progress_text, existing_eval,
                           factual_score, factual_justification, completeness_score, 
                           completeness_justification, clarity_score, clarity_justification])
    
    # Load initial data
    demo.load(update_display, outputs=[predicted_label, top5_shap, word_counter, explanation, shap_values, meta_desc, 
                                      progress_text, existing_eval,
                                      factual_score, factual_justification, completeness_score, 
                                      completeness_justification, clarity_score, clarity_justification])

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
