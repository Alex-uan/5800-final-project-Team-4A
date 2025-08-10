# SMS Spam Detection System

A comprehensive multi-class SMS spam detection system using machine learning to identify and classify different types of spam messages. The system features both a modern GUI interface and a command-line interface for real-time SMS message analysis.

## Features

- **Multi-class Classification**: Detects 14+ different types of spam including:
  - Prize/Lottery Scams
  - Romance Scams
  - Free Service Scams
  - Business Executive Spoofs
  - Subscription Scams
  - And more...

- **Advanced Feature Engineering**:
  - 14 hand-crafted linguistic features
  - 25 TF-IDF text features
  - Composite scam pattern scoring

- **Machine Learning**:
  - Random Forest classifier with balanced sampling
  - Cross-validation and performance evaluation
  - Confidence scores and probability rankings

- **Safety Features**:
  - Detailed safety advice for each spam type
  - Real-time analysis with instant results
  - Debug mode for feature analysis

## Requirements

- Python 3.7 or higher
- Required packages (see requirements.txt)
- Dataset file: `balanced_spam_data 2.0.csv`

## Installation

1. **Clone or download the project files**:
   ```
   sms_spam_detector.py
   sms_gui.py
   main.py
   requirements.txt
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**:
   - Place the `balanced_spam_data 2.0.csv` file on your Desktop
   - The dataset should contain columns: `messages`, `label`, and `Type`

## Dataset Format

Your CSV file should have the following structure:

```csv
messages,label,Type
"Free ringtone! Text NOW to 12345",spam,Free_Service_Scam
"Hi, how are you?",ham,
"CONGRATULATIONS! You won $1000!",spam,Prize_Lottery_Scam
```

**Required columns**:
- `messages`: SMS message text
- `label`: Either "spam" or "ham" (legitimate)
- `Type`: (Optional) Specific spam type for spam messages

## Usage

### Quick Start

Run the main program:
```bash
python main.py
```

The system will:
1. Load and process your dataset
2. Engineer features and train the model
3. Evaluate model performance
4. Prompt you to choose an interface

### Interface Options

#### 1. GUI Mode (Recommended)
- User-friendly interface
- Real-time message analysis
- Detailed results with safety advice
- Easy message input and analysis

#### 2. Terminal Mode
- Command-line interface
- Debug capabilities
- Interactive commands:
  - Enter any message for detection
  - `debug MESSAGE` - Detailed feature analysis
  - `examples` - View test examples
  - `help` - Show help information
  - `quit` - Exit program

### Command Line Options

```bash
python main.py           # Run full system
python main.py --test    # Quick test mode
python main.py --info    # Show system information
python main.py --help    # Show help
```

### Example Usage

**GUI Mode**:
1. Run `python main.py`
2. Choose option 1 for GUI
3. Enter an SMS message in the text box
4. Click "Detect" or press Enter
5. View detailed results and safety advice
6. Click "Analyze Another Message" to continue

**Terminal Mode**:
```
Enter SMS message to detect: URGENT! You won £1000! Call now!
Analyzing...

DETECTION RESULTS
============================================================
Input message: "URGENT! You won £1000! Call now!"
Message length: 35 characters

RESULT: SPAM DETECTED
Spam type: Prize_Lottery_Scam
Confidence: 89.2%
Safety advice: Never call numbers or reply to prize messages
```

## Model Performance

The system uses a Random Forest classifier optimized for datasets with ~1750 records:

- **Features**: 39 total features (14 hand-crafted + 25 TF-IDF)
- **Cross-validation**: 5-fold CV with F1-score evaluation
- **Class balancing**: Handles imbalanced datasets
- **Confidence scoring**: Provides prediction confidence levels

## File Structure

```
├── main.py                    # Main driver program
├── sms_spam_detector.py       # Core detection logic and ML model
├── sms_gui.py                 # GUI interface using tkinter
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── balanced_spam_data 2.0.csv # Dataset (place on Desktop)
```

## Architecture

### Core Components

1. **MultiClassSMSDetector**: Main detection class
   - Data loading and preprocessing
   - Feature engineering and extraction
   - Model training and evaluation
   - Prediction and analysis methods

2. **SMSDetectorTkinterGUI**: GUI interface
   - Input interface for message entry
   - Results display with detailed analysis
   - Navigation between screens

3. **Main Driver**: Coordination and workflow
   - System initialization
   - Model training pipeline
   - Interface selection and launch

### Feature Engineering

**Hand-crafted Features**:
- Text length and word count
- Character ratios (digits, uppercase)
- Punctuation analysis
- Keyword-based scoring (urgency, money terms, etc.)
- Phone number detection
- Composite scam pattern scoring

**TF-IDF Features**:
- 25 most important n-grams (1-2 grams)
- English stop words removal
- Sublinear term frequency scaling

## Troubleshooting

### Common Issues

1. **Dataset not found**:
   - Ensure `balanced_spam_data 2.0.csv` is on your Desktop
   - Check file name spelling exactly

2. **Import errors**:
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version (3.7+ required)

3. **GUI not displaying**:
   - Ensure tkinter is installed (usually comes with Python)
   - Try terminal mode as alternative

4. **Low accuracy**:
   - Check dataset quality and format
   - Ensure sufficient samples per class (minimum 5)

### Error Messages

- `"Data file not found!"`: Place dataset on Desktop
- `"Please train the model first!"`: Run full training pipeline
- `"Need at least 2 classes"`: Check dataset has both spam and ham samples


## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify your dataset format
3. Ensure all dependencies are installed
4. Use debug mode for detailed analysis

## Contributing

To contribute:
1. Follow the existing code structure
2. Add comprehensive docstrings
3. Test with various message types
4. Update documentation as needed
