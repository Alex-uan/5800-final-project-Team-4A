#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMS Spam Detection GUI Interface
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import warnings
warnings.filterwarnings('ignore')

class SMSDetectorTkinterGUI:
    def __init__(self, detector):
        """
        Initialize SMS spam detection GUI using tkinter
        Args:
            detector: Trained MultiClassSMSDetector instance
        """
        self.detector = detector
        self.setup_interface()
        
    def setup_interface(self):
        """Create clean tkinter interface"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("SMS Spam Detection System")
        self.root.geometry("950x950")
        self.root.configure(bg='#f8f9fa')
        
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="SMS Spam Detection System", 
            font=("Arial", 24, "bold"),
            bg='#f8f9fa',
            fg='#1a1a1a'
        )
        title_label.grid(row=0, column=0, pady=20)
        
        # Create main frame that will switch between input and results
        self.main_frame = tk.Frame(self.root, bg='#f8f9fa')
        self.main_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.show_input_interface()
        
    def show_input_interface(self):
        """Show the input interface"""
        # Clear main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Input frame
        input_frame = tk.Frame(self.main_frame, bg='#f8f9fa')
        input_frame.grid(row=0, column=0, sticky='ew')
        input_frame.grid_columnconfigure(1, weight=1)
        
        # Input label
        input_label = tk.Label(
            input_frame,
            text="Enter SMS message:",
            font=("Arial", 14, "bold"),
            bg='#f8f9fa',
            fg='#1a1a1a'
        )
        input_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Text input
        self.text_input = tk.Entry(
            input_frame,
            font=("Arial", 12),
            relief='solid',
            bd=2,
            bg='white',
            fg='#1a1a1a'
        )
        self.text_input.grid(row=1, column=0, columnspan=2, sticky='ew', padx=(0, 10))
        self.text_input.bind('<Return>', self.analyze_message)
        
        # Detect button
        self.detect_btn = tk.Button(
            input_frame,
            text="Detect",
            font=("Arial", 12, "bold"),
            bg='#17a2b8',
            fg='#1e40af',
            relief='flat',
            command=self.analyze_message,
            cursor='hand2'
        )
        self.detect_btn.grid(row=1, column=2)
        
        # Welcome text frame
        welcome_frame = tk.Frame(self.main_frame, bg='white', relief='solid', bd=1)
        welcome_frame.grid(row=1, column=0, sticky='nsew', pady=20)
        welcome_frame.grid_rowconfigure(0, weight=1)
        welcome_frame.grid_columnconfigure(0, weight=1)
        
        # Welcome text
        welcome_text = scrolledtext.ScrolledText(
            welcome_frame,
            font=("Arial", 11),
            bg='white',
            fg='#1a1a1a',
            wrap=tk.WORD,
            state='disabled'
        )
        welcome_text.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # Show welcome message
        welcome_text.config(state='normal')
        welcome_text.insert(tk.END, """SMS Spam Detection System

Enter any SMS message in the text box above and click 'Detect' to analyze.

Examples:
• "Hi, how are you doing today?"
• "CONGRATULATIONS! You won $1,000,000!"
• "URGENT: Your account has been suspended"
• "Free ringtone! Text NOW to 12345"

Enter a message above to get started.""")
        welcome_text.config(state='disabled')
        
        # Focus on input
        self.text_input.focus()
    
    def show_results_interface(self, message, result):
        """Show the results interface"""
        # Clear main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Results frame
        results_frame = tk.Frame(self.main_frame, bg='white', relief='solid', bd=1)
        results_frame.grid(row=0, column=0, sticky='nsew')
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Results text area
        results_text = scrolledtext.ScrolledText(
            results_frame,
            font=("Arial", 11),
            bg='white',
            fg='#1a1a1a',
            wrap=tk.WORD,
            state='disabled'
        )
        results_text.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # Display results
        results_text.config(state='normal')
        
        # Message preview
        msg_preview = message[:80] + "..." if len(message) > 80 else message
        result_text = f"Message: \"{msg_preview}\"\n\n"
        
        # Main result
        if result['is_spam']:
            result_text += "🚨 SPAM DETECTED\n"
            result_text += "=" * 50 + "\n\n"
            
            # Spam type
            spam_type = result['spam_type'].replace('_', ' ')
            result_text += f"Type: {spam_type}\n\n"
            
            # Safety warning
            result_text += "Safety Warning:\n"
            advice = self.get_safety_advice(result['spam_type'])
            result_text += f"{advice}\n\n"
            
        else:
            result_text += "✅ SAFE MESSAGE\n"
            result_text += "=" * 50 + "\n\n"
            result_text += "This message appears to be legitimate.\n\n"
        
        # Confidence score
        confidence = result['confidence']
        if confidence > 0.8:
            conf_text = "High Confidence"
        elif confidence > 0.6:
            conf_text = "Medium Confidence"
        else:
            conf_text = "Low Confidence"
        
        result_text += f"Confidence: {confidence:.1%} ({conf_text})\n\n"
        
        # Top predictions
        result_text += "Top Predictions:\n"
        result_text += "-" * 50 + "\n"
        
        top_probs = sorted(result['all_probabilities'].items(), 
                          key=lambda x: x[1], reverse=True)[:3]
        
        for i, (class_name, prob) in enumerate(top_probs, 1):
            result_text += f"{i}. {class_name}: {prob:.1%}\n"
        
        results_text.insert(tk.END, result_text)
        results_text.config(state='disabled')
        
        # Back button frame
        button_frame = tk.Frame(self.main_frame, bg='#f8f9fa')
        button_frame.grid(row=1, column=0, sticky='ew', pady=10)
        
        # Back button
        back_btn = tk.Button(
            button_frame,
            text="Analyze Another Message",
            font=("Arial", 12, "bold"),
            bg='#17a2b8',
            fg='#1e40af',
            relief='flat',
            command=self.show_input_interface,
            cursor='hand2'
        )
        back_btn.pack()
    
    def analyze_message(self, event=None):
        """Analyze the input message"""
        message = self.text_input.get().strip()
        
        if not message:
            self.show_error("Please enter a message to analyze")
            return
        
        try:
            # Get prediction
            result = self.detector.predict_message_type(message)
            
            if result is None:
                self.show_error("Analysis failed. Please try again.")
                return
            
            # Show results interface
            self.show_results_interface(message, result)
            
        except Exception as e:
            print(f"Error: {e}")
            self.show_error("An error occurred during analysis")
    
    def show_error(self, error_message):
        """Display error message in results interface"""
        # Clear main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        # Error frame
        error_frame = tk.Frame(self.main_frame, bg='white', relief='solid', bd=1)
        error_frame.grid(row=0, column=0, sticky='nsew')
        error_frame.grid_rowconfigure(0, weight=1)
        error_frame.grid_columnconfigure(0, weight=1)
        
        # Error text
        error_text = scrolledtext.ScrolledText(
            error_frame,
            font=("Arial", 11),
            bg='white',
            fg='#1a1a1a',
            wrap=tk.WORD,
            state='disabled'
        )
        error_text.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        error_text.config(state='normal')
        error_text.insert(tk.END, f"Error\n\n{error_message}")
        error_text.config(state='disabled')
        
        # Back button frame
        button_frame = tk.Frame(self.main_frame, bg='#f8f9fa')
        button_frame.grid(row=1, column=0, sticky='ew', pady=10)
        
        # Back button
        back_btn = tk.Button(
            button_frame,
            text="Go Back",
            font=("Arial", 12, "bold"),
            bg='#17a2b8',
            fg='#1e40af',
            relief='flat',
            command=self.show_input_interface,
            cursor='hand2'
        )
        back_btn.pack()
    
    def get_safety_advice(self, spam_type):
        """Get safety advice for spam type"""
        advice_dict = {
            'Prize_Lottery_Scam': "Never call numbers or reply to prize messages. Real prizes don't require payment.",
            'Romance_Scam': "Beware of emotional manipulation. Never send money to someone you haven't met.",
            'Free_Service_Scam': "Nothing is truly free. Watch for hidden fees and subscription traps.",
            'Contact_Service_Scam': "Don't call unknown numbers. These often lead to premium rate charges.",
            'Subscription_Scam': "Don't reply - may auto-enroll you in expensive services.",
            'Business_Executive_Spoofs': "Verify through official channels. Don't transfer money without verification.",
            'Card_not_present': "Never provide financial details to unsolicited messages.",
            'Package_Delivery_Phishing_Scam': "Verify with official courier websites, not links in messages.",
            'Marketing_Link_Scam': "Don't click suspicious links. They may contain malware.",
            'spam_unknown': "Handle with extreme caution. Don't click links or reply."
        }
        return advice_dict.get(spam_type, "This is spam. Handle with caution and don't provide personal information.")
    
    def show(self):
        """Show the interface"""
        self.root.mainloop()

if __name__ == "__main__":
    print("SMS GUI module loaded!")
    print("This module should be imported, not run directly.")
    print("Use: from sms_gui import SMSDetectorTkinterGUI")