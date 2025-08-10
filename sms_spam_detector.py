#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Class SMS Spam Detection System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import warnings
warnings.filterwarnings('ignore')

class MultiClassSMSDetector:
    def __init__(self, csv_file_path):
        """
        Initialize SMS spam detector with dataset path
        
        Args:
            csv_file_path (str): Path to the CSV dataset file
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_model = None
        self.tfidf_vectorizer = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """
        Load SMS dataset from CSV file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.data = pd.read_csv(self.csv_file_path)
            print(f"Data loaded: {len(self.data)} records")
            print(f"Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"Failed to load data: {e}")
            return False
    
    def create_multiclass_labels(self):
        """
        Create multi-class labels from spam types
        
        Returns:
            int: Number of unique classes created
        """
        print("Creating multi-class labels...")
        
        self.data['multiclass_label'] = self.data['label'].copy()
        spam_mask = self.data['label'] == 'spam'
        
        if 'Type' in self.data.columns:
            spam_with_type = spam_mask & self.data['Type'].notna() & (self.data['Type'] != '')
            self.data.loc[spam_with_type, 'multiclass_label'] = self.data.loc[spam_with_type, 'Type']
            
            spam_without_type = spam_mask & (self.data['Type'].isna() | (self.data['Type'] == ''))
            self.data.loc[spam_without_type, 'multiclass_label'] = 'spam_unknown'
        
        label_counts = self.data['multiclass_label'].value_counts()
        print(f"Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Filter classes with minimum samples
        min_samples = 5
        valid_labels = label_counts[label_counts >= min_samples].index
        original_count = len(self.data)
        self.data = self.data[self.data['multiclass_label'].isin(valid_labels)]
        
        print(f"After filtering (min {min_samples} samples): {len(self.data)} samples")
        final_counts = self.data['multiclass_label'].value_counts()
        print(f"Final classes: {len(final_counts)}")
        
        return len(final_counts)
    
    def preprocess_text(self, text):
        """
        Basic text preprocessing for SMS messages
        
        Args:
            text (str): Raw SMS text
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)  # Keep alphanumeric and basic punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate very long messages
        if len(text) > 500:
            text = text[:500]
        
        return text
    
    def extract_basic_features(self, texts):
        """
        Extract hand-crafted features from text messages
        
        Args:
            texts (list): List of SMS message texts
            
        Returns:
            dict: Dictionary of feature arrays
        """
        features = {
            'length': [], 'word_count': [], 'digit_ratio': [], 'uppercase_ratio': [],
            'exclamation_count': [], 'urgency_score': [], 'money_terms': [], 'phone_numbers': [],
            'call_to_action_score': [], 'all_caps_words': [], 'free_terms': [], 'authority_appeal': [],
            'prize_bonus_score': [], 'scam_pattern_score': []
        }
        
        # Define keyword lists for different spam indicators
        urgency_words = ['urgent', 'immediately', 'now', 'asap', 'hurry', 'quick', 'fast', 'today']
        money_words = ['win', 'won', 'prize', 'money', 'dollar', 'million', 'billion', 'cash', 'reward', '$', '£', '€', 'pounds']
        action_words = ['click', 'call', 'text', 'reply', 'redeem', 'claim', 'download', 'visit']
        free_words = ['free', 'gratis', 'bonus', 'gift', 'complimentary']
        authority_words = ['bank', 'official', 'security', 'account', 'government']
        prize_words = ['congratulations', 'congrats', 'winner', 'selected', 'won', 'prize', 'lottery', 'jackpot', 'draw', 'competition']
        
        for text in texts:
            original_text = str(text) if not pd.isna(text) else ""
            processed_text = original_text.lower()
            
            # Basic text statistics
            features['length'].append(len(original_text))
            
            words = processed_text.split()
            features['word_count'].append(len(words))
            
            # Character ratios
            if original_text:
                features['digit_ratio'].append(sum(1 for c in original_text if c.isdigit()) / len(original_text))
                features['uppercase_ratio'].append(sum(1 for c in original_text if c.isupper()) / len(original_text))
            else:
                features['digit_ratio'].append(0)
                features['uppercase_ratio'].append(0)
            
            # Punctuation and formatting features
            features['exclamation_count'].append(original_text.count('!'))
            features['all_caps_words'].append(len(re.findall(r'\b[A-Z]{2,}\b', original_text)))
            
            # Keyword-based features
            features['urgency_score'].append(sum(1 for word in urgency_words if word in processed_text))
            features['money_terms'].append(sum(1 for word in money_words if word in processed_text))
            features['call_to_action_score'].append(sum(1 for word in action_words if word in processed_text))
            features['free_terms'].append(sum(1 for word in free_words if word in processed_text))
            features['authority_appeal'].append(sum(1 for word in authority_words if word in processed_text))
            
            # Phone number detection
            features['phone_numbers'].append(len(re.findall(r'\b\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b', processed_text)))
            
            # Prize/lottery specific score
            prize_score = sum(2 if word in processed_text else 0 for word in prize_words)
            features['prize_bonus_score'].append(prize_score)
            
            # Composite scam pattern score
            scam_score = (
                features['money_terms'][-1] * 3 +
                features['urgency_score'][-1] * 2 +
                features['exclamation_count'][-1] +
                features['all_caps_words'][-1] * 2 +
                features['call_to_action_score'][-1] * 2 +
                prize_score
            )
            features['scam_pattern_score'].append(scam_score)
        
        return features
    
    def prepare_features(self, text_column, target_column):
        """
        Prepare features for training by combining hand-crafted and TF-IDF features
        
        Args:
            text_column (str): Name of text column in dataset
            target_column (str): Name of target column in dataset
            
        Returns:
            bool: True if successful, False otherwise
        """
        print("Starting feature engineering...")
        
        num_classes = self.create_multiclass_labels()
        if num_classes < 2:
            print("Error: Need at least 2 classes for classification")
            return False
        
        # Preprocess text data
        self.data['processed_text'] = self.data[text_column].apply(self.preprocess_text)
        
        # Create TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=25, ngram_range=(1, 2), stop_words='english',
            min_df=4, max_df=0.75, sublinear_tf=True,
            lowercase=True, token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(self.data['processed_text'])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Extract hand-crafted features
        text_features = self.extract_basic_features(self.data[text_column])
        text_features_df = pd.DataFrame(text_features)
        
        # Combine all features
        all_features = pd.concat([text_features_df, tfidf_df], axis=1)
        
        print(f"Feature engineering completed:")
        print(f"  Hand-crafted features: {text_features_df.shape[1]}")
        print(f"  TF-IDF features: {tfidf_df.shape[1]}")
        print(f"  Total features: {all_features.shape[1]}")
        
        # Encode labels and split data
        y = self.label_encoder.fit_transform(self.data['multiclass_label'])
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            all_features, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return True
    
    def train_multiclass_model(self):
        """
        Train Random Forest classifier for multi-class spam detection
        
        Returns:
            RandomForestClassifier: Trained model
        """
        print("Training multi-class Random Forest model...")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', random_state=42, bootstrap=True, oob_score=True,
            class_weight='balanced_subsample', criterion='gini', max_samples=0.8
        )
        
        self.rf_model.fit(self.X_train, self.y_train)
        
        print(f"Out-of-bag score: {self.rf_model.oob_score_:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.rf_model, self.X_train, self.y_train, cv=5, scoring='f1_weighted')
        print(f"5-fold CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.rf_model
    
    def evaluate_multiclass_model(self):
        """
        Evaluate model performance on test set
        
        Returns:
            tuple: (accuracy, predictions, probabilities)
        """
        if self.rf_model is None:
            print("Please train the model first!")
            return
        
        y_pred = self.rf_model.predict(self.X_test)
        y_pred_proba = self.rf_model.predict_proba(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nModel Evaluation:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        class_names = self.label_encoder.classes_
        report = classification_report(
            self.y_test, y_pred, target_names=class_names, zero_division=0
        )
        print(f"\nClassification Report:")
        print(report)
        
        return accuracy, y_pred, y_pred_proba

    def predict_message_type(self, sms_text):
        """
        Predict spam type for a given SMS message
        
        Args:
            sms_text (str): SMS message text to analyze
            
        Returns:
            dict: Prediction results containing spam status, type, confidence, and probabilities
        """
        if self.rf_model is None:
            print("Please train the model first!")
            return None
        
        # Preprocess input text
        processed_text = self.preprocess_text(sms_text)
        
        # Extract features
        text_features = self.extract_basic_features([sms_text])
        text_features_df = pd.DataFrame(text_features)
        
        tfidf_features = self.tfidf_vectorizer.transform([processed_text])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Combine features and align with training features
        features = pd.concat([text_features_df, tfidf_df], axis=1)
        features = features.reindex(columns=self.X_train.columns, fill_value=0)
        
        # Make prediction
        prediction = self.rf_model.predict(features)[0]
        probabilities = self.rf_model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary for all classes
        all_classes = self.label_encoder.classes_
        class_probabilities = {
            class_name: prob 
            for class_name, prob in zip(all_classes, probabilities)
        }
        
        is_spam = predicted_label != 'ham'
        spam_type = predicted_label if is_spam else None
        
        return {
            'is_spam': is_spam,
            'spam_type': spam_type,
            'predicted_class': predicted_label,
            'confidence': confidence,
            'all_probabilities': class_probabilities
        }

    def debug_prediction(self, sms_text):
        """
        Debug prediction with detailed feature analysis
        
        Args:
            sms_text (str): SMS message text to debug
            
        Returns:
            dict: Prediction results with debug information
        """
        if self.rf_model is None:
            print("Please train the model first!")
            return None
        
        print(f"\nDEBUG: Prediction Analysis")
        print(f"Input text: \"{sms_text}\"")
        
        # Extract and analyze features
        text_features = self.extract_basic_features([sms_text])
        
        print(f"\nFeature Values:")
        spam_indicators = 0
        for feature_name, values in text_features.items():
            value = values[0]
            print(f"{feature_name}: {value}")
            
            # Identify potential spam indicators
            if feature_name == 'money_terms' and value > 0:
                print(f"  Money/prize terms detected!")
                spam_indicators += 3
            if feature_name == 'exclamation_count' and value > 1:
                print(f"  Excessive punctuation detected!")
                spam_indicators += 1
            if feature_name == 'all_caps_words' and value > 0:
                print(f"  All-caps words detected!")
                spam_indicators += 2
            if feature_name == 'urgency_score' and value > 0:
                print(f"  Urgency language detected!")
                spam_indicators += 2
            if feature_name == 'scam_pattern_score' and value > 5:
                print(f"  High scam score!")
                spam_indicators += 3
        
        print(f"\nTotal spam indicators: {spam_indicators}")
        
        # Get model prediction
        result = self.predict_message_type(sms_text)
        
        print(f"\nModel Prediction:")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Is spam: {result['is_spam']}")
        
        print(f"\nAll Class Probabilities:")
        for class_name, prob in sorted(result['all_probabilities'].items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"{class_name}: {prob:.1%}")
        
        return result

    def interactive_detection(self):
        """Interactive terminal mode for spam detection"""
        print("\n" + "="*60)
        print("INTERACTIVE SPAM DETECTION MODE")
        print("="*60)
        print("Commands:")
        print("  Enter any text message for detection")
        print("  'quit', 'exit', 'q' to exit")
        print("  'help' for help")
        print("  'examples' to see examples")
        print("  'debug MESSAGE' for detailed analysis")
        print("-"*60)
        
        while True:
            try:
                print()
                user_input = input("Enter SMS message to detect: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Thank you for using SMS Spam Detection System!")
                    break
                
                if user_input.lower().startswith('debug '):
                    debug_message = user_input[6:].strip()
                    if debug_message:
                        self.debug_prediction(debug_message)
                    else:
                        print("Please provide a message after 'debug'.")
                    continue
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'examples':
                    self.show_examples()
                    continue
                
                if not user_input:
                    print("Please enter a valid text message")
                    continue
                
                print("Analyzing...")
                result = self.predict_message_type(user_input)
                
                if result is None:
                    print("Prediction failed, please ensure model is properly trained")
                    continue
                
                self.display_terminal_result(user_input, result)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error occurred: {e}")
    
    def show_help(self):
        """Display help information"""
        print("\nHELP - SMS Spam Detection")
        print("-" * 40)
        print("Commands:")
        print("  Enter text directly - perform detection")
        print("  'debug MESSAGE' - detailed feature analysis") 
        print("  'examples' - view test examples")
        print("  'help' - show this help")
        print("  'quit' - exit program")
        print("-" * 40)
    
    def show_examples(self):
        """Display example messages for testing"""
        print("\nEXAMPLES - Test examples")
        print("-" * 40)
        
        examples = [
            "Congrats!! YOU WON 2 MILLION DOLLARS!! Redeem it NOW",
            "URGENT! You have won £1000! Call 09061234567 now!",
            "Hi, how are you doing today?",
            "FREE ringtone! Text NOW to 12345",
            "Hello sweetie, I'm Sarah from dating agency",
            "Your account has been suspended. Click link",
            "Thanks for the birthday wishes!",
            "CONGRATULATIONS! You've won a free iPhone!"
        ]
        
        print("You can copy the following examples for testing:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        
        print("\nExample debug usage:")
        print("debug Congrats!! YOU WON 2 MILLION DOLLARS!! Redeem it NOW")
        print("-" * 40)
    
    def display_terminal_result(self, message, result):
        """
        Display results in terminal format
        
        Args:
            message (str): Original input message
            result (dict): Prediction results
        """
        print("\n" + "="*60)
        print("DETECTION RESULTS")
        print("="*60)
        
        print(f"Input message: \"{message}\"")
        print(f"Message length: {len(message)} characters")
        
        if result['is_spam']:
            print(f"\nRESULT: SPAM DETECTED")
            print(f"Spam type: {result['spam_type']}")
            print(f"Confidence: {result['confidence']:.1%}")
            self.give_advice(result['spam_type'])
        else:
            print(f"\nRESULT: LEGITIMATE MESSAGE")
            print(f"Confidence: {result['confidence']:.1%}")
            print("This message appears to be safe.")
        
        # Show top predictions
        print(f"\nTop predictions:")
        top_probs = sorted(result['all_probabilities'].items(), 
                          key=lambda x: x[1], reverse=True)[:3]
        
        for i, (class_name, prob) in enumerate(top_probs, 1):
            marker = "-> " if class_name == result['predicted_class'] else "   "
            print(f"  {i}. {marker}{class_name}: {prob:.1%}")
        
        print("="*60)
    
    def give_advice(self, spam_type):
        """
        Provide safety advice for detected spam type
        
        Args:
            spam_type (str): Type of spam detected
        """
        advice_dict = {
            'Prize_Lottery_Scam': "Never call numbers or reply to prize messages",
            'Romance_Scam': "Beware of emotional manipulation from strangers",
            'Free_Service_Scam': "Nothing is truly free, watch for hidden fees",
            'Contact_Service_Scam': "Don't call unknown numbers, may incur charges",
            'Subscription_Scam': "Don't reply - may auto-subscribe to paid services",
            'Business_Executive_Spoofs': "Verify sender identity, don't transfer money",
            'Card_not_present': "Don't provide credit card info to strangers",
            'spam_unknown': "Handle with caution, don't click links or reply"
        }
        
        advice = advice_dict.get(spam_type, "This is spam, handle with caution")
        print(f"Safety advice: {advice}")

def main():
    """Main function for standalone execution"""
    print("SMS Spam Detection System")
    
    import os
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    csv_file = os.path.join(desktop_path, "balanced_spam_data 2.0.csv")
    
    detector = MultiClassSMSDetector(csv_file)
    
    if not detector.load_data():
        return
    
    # Clean data
    text_column = "messages"
    target_column = "multiclass_label"
    detector.data = detector.data.dropna(subset=[text_column, 'label'])
    print(f"Clean data: {len(detector.data)} records")
    
    # Prepare features and train model
    if not detector.prepare_features(text_column, target_column):
        return
    
    detector.train_multiclass_model()
    accuracy, _, _ = detector.evaluate_multiclass_model()
    
    print(f"\nModel accuracy: {accuracy:.4f}")
    
    # Choose interface mode
    print("\nChoose interface mode:")
    print("1. GUI mode")
    print("2. Terminal mode")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            print("Starting GUI...")
            # GUI would be imported and started here
        else:
            detector.interactive_detection()
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()