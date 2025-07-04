"""
Generate sample training data for DistillBERT sentiment analysis
Creates train.csv, validation.csv, and test.csv files
"""

import pandas as pd
import numpy as np
import random
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_sample_data():
    """Generate comprehensive sample data for sentiment analysis"""

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Sample positive texts
    positive_texts = [
        "I absolutely love this product! It exceeded my expectations.",
        "Amazing quality and fast delivery. Highly recommended!",
        "Great experience with excellent customer service.",
        "Fantastic product that works perfectly as described.",
        "Outstanding value for money. Very satisfied with my purchase.",
        "Excellent performance and reliable functionality.",
        "Perfect solution for my needs. Couldn't be happier!",
        "Incredible results and user-friendly interface.",
        "Superb customer support and quick response times.",
        "Highly satisfied with the quality and durability.",
        "Wonderful product that delivers on its promises.",
        "Impressive features and seamless integration.",
        "Great investment, worth every penny spent.",
        "Exceptional quality control and attention to detail.",
        "Brilliant design and intuitive user experience.",
        "Top-notch performance with consistent results.",
        "Remarkable improvement in my workflow efficiency.",
        "Outstanding build quality and premium materials.",
        "Excellent documentation and easy setup process.",
        "Fantastic support team that goes above and beyond.",
        "Love the sleek design and modern aesthetics.",
        "Perfect functionality with no issues whatsoever.",
        "Great value proposition for the price point.",
        "Impressive innovation and cutting-edge technology.",
        "Excellent reliability and consistent performance.",
        "Amazing features that enhance productivity significantly.",
        "Wonderful user interface with intuitive navigation.",
        "Great product that solves real problems effectively.",
        "Fantastic build quality with premium finish.",
        "Excellent performance optimization and speed.",
        "Love the comprehensive feature set available.",
        "Perfect integration with existing systems.",
        "Outstanding customer experience from start to finish.",
        "Great documentation with clear instructions.",
        "Fantastic community support and active development.",
        "Excellent security features and data protection.",
        "Love the customization options available.",
        "Perfect balance of features and simplicity.",
        "Outstanding technical support and expertise.",
        "Great scalability for future growth needs.",
        "Excellent mobile app with smooth performance.",
        "Love the real-time updates and notifications.",
        "Perfect for both beginners and advanced users.",
        "Outstanding analytics and reporting capabilities.",
        "Great API documentation and developer resources.",
        "Excellent backup and recovery features.",
        "Love the collaborative features for team work.",
        "Perfect automation capabilities that save time.",
        "Outstanding integration with third-party tools.",
        "Great flexibility in configuration options."
    ]

    # Sample negative texts
    negative_texts = [
        "Terrible product quality, completely disappointed.",
        "Very poor customer service and slow response times.",
        "Waste of money, doesn't work as advertised.",
        "Extremely disappointed with the overall experience.",
        "Poor build quality and frequent malfunctions.",
        "Completely unsatisfied with this purchase decision.",
        "Worst product I've ever bought, total regret.",
        "Defective item received with missing components.",
        "Misleading description and poor actual performance.",
        "Rude customer service staff and unhelpful attitude.",
        "Overpriced for the poor quality delivered.",
        "Broken on arrival with damaged packaging.",
        "Doesn't meet basic functionality requirements.",
        "Poor design choices and confusing interface.",
        "Unreliable performance with frequent crashes.",
        "Terrible user experience and frustrating bugs.",
        "Poor documentation with unclear instructions.",
        "Slow performance that affects productivity negatively.",
        "Inadequate features for the price paid.",
        "Poor compatibility with existing systems.",
        "Terrible installation process with multiple errors.",
        "Unreliable software with data loss issues.",
        "Poor security implementation with vulnerabilities.",
        "Terrible mobile app that crashes frequently.",
        "Inadequate support resources and knowledge base.",
        "Poor update process that breaks existing functionality.",
        "Terrible backup system that fails regularly.",
        "Unreliable notifications and alert system.",
        "Poor integration capabilities with other tools.",
        "Terrible API documentation and examples.",
        "Inadequate customization options available.",
        "Poor scalability for growing business needs.",
        "Terrible analytics with inaccurate reporting.",
        "Unreliable cloud service with frequent downtime.",
        "Poor user management and permission system.",
        "Terrible search functionality that doesn't work.",
        "Inadequate automation features for efficiency.",
        "Poor export options and data portability.",
        "Terrible collaboration tools for team work.",
        "Unreliable email notifications and alerts.",
        "Poor dashboard design and information display.",
        "Terrible loading times and performance issues.",
        "Inadequate training resources and tutorials.",
        "Poor data validation and error handling.",
        "Terrible user onboarding experience.",
        "Unreliable third-party integrations.",
        "Poor accessibility features for users.",
        "Terrible workflow management capabilities.",
        "Inadequate reporting and analytics features.",
        "Poor multi-language support and localization."
    ]

    # Sample neutral texts
    neutral_texts = [
        "The product is okay, nothing special but functional.",
        "Average quality for the price point offered.",
        "It works as expected, no major issues.",
        "Standard features with basic functionality provided.",
        "Mediocre performance but gets the job done.",
        "Regular product with typical characteristics.",
        "Acceptable quality for everyday use cases.",
        "Basic functionality without advanced features.",
        "Average user interface with standard design.",
        "Typical customer service experience received.",
        "Standard delivery time and packaging quality.",
        "Regular performance with expected results.",
        "Basic documentation provided with product.",
        "Average build quality for this price range.",
        "Standard installation process without complications.",
        "Typical software with common features.",
        "Average mobile app functionality provided.",
        "Standard integration capabilities offered.",
        "Regular update schedule and maintenance.",
        "Basic customization options available.",
        "Average analytics and reporting features.",
        "Standard security measures implemented.",
        "Typical backup and recovery options.",
        "Regular notification system functionality.",
        "Basic collaboration features provided.",
        "Average automation capabilities offered.",
        "Standard export and import options.",
        "Typical user management system.",
        "Basic search functionality implemented.",
        "Average dashboard design and layout.",
        "Standard loading times and performance.",
        "Regular training materials provided.",
        "Basic data validation features.",
        "Average onboarding process experience.",
        "Standard third-party integrations available.",
        "Regular accessibility features implemented.",
        "Basic workflow management capabilities.",
        "Average multi-language support provided.",
        "Standard API documentation quality.",
        "Regular community support available."
    ]

    # Create comprehensive dataset
    def create_dataset(size):
        """Create a balanced dataset with specified size"""
        # Calculate samples per class (roughly equal distribution)
        positive_samples = size // 3
        negative_samples = size // 3
        neutral_samples = size - positive_samples - negative_samples

        texts = []
        labels = []

        # Add positive samples
        for i in range(positive_samples):
            text = random.choice(positive_texts)
            # Add some variation to avoid exact duplicates
            if random.random() < 0.3:  # 30% chance to add variation
                variations = [
                    f"Really {text.lower()}",
                    f"{text} Amazing!",
                    f"I think {text.lower()}",
                    f"Definitely {text.lower()}",
                    f"Absolutely {text.lower()}"
                ]
                text = random.choice(variations)
            texts.append(text)
            labels.append(1)  # Positive

        # Add negative samples
        for i in range(negative_samples):
            text = random.choice(negative_texts)
            # Add some variation
            if random.random() < 0.3:  # 30% chance to add variation
                variations = [
                    f"Really {text.lower()}",
                    f"{text} Unfortunately.",
                    f"I think {text.lower()}",
                    f"Definitely {text.lower()}",
                    f"Absolutely {text.lower()}"
                ]
                text = random.choice(variations)
            texts.append(text)
            labels.append(0)  # Negative

        # Add neutral samples (mix of positive and negative labels)
        for i in range(neutral_samples):
            text = random.choice(neutral_texts)
            # Add some variation
            if random.random() < 0.3:  # 30% chance to add variation
                variations = [
                    f"I think {text.lower()}",
                    f"Generally, {text.lower()}",
                    f"Overall, {text.lower()}",
                    f"In my opinion, {text.lower()}"
                ]
                text = random.choice(variations)
            texts.append(text)
            # Neutral texts get mixed labels (some positive, some negative)
            labels.append(random.choice([0, 1]))

        # Shuffle the dataset
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)

        return list(texts), list(labels)

    # Generate datasets
    print("Generating training data...")
    train_texts, train_labels = create_dataset(5000)

    print("Generating validation data...")
    val_texts, val_labels = create_dataset(1000)

    print("Generating test data...")
    test_texts, test_labels = create_dataset(1000)

    # Create DataFrames
    train_df = pd.DataFrame({
        'text': train_texts,
        'label': train_labels
    })

    val_df = pd.DataFrame({
        'text': val_texts,
        'label': val_labels
    })

    test_df = pd.DataFrame({
        'text': test_texts,
        'label': test_labels
    })

    # Save to CSV files
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/validation.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    # Print statistics
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    print(f"\nTraining Dataset:")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Positive samples: {sum(train_df['label'])}")
    print(f"  Negative samples: {len(train_df) - sum(train_df['label'])}")
    print(f"  Positive ratio: {sum(train_df['label']) / len(train_df) * 100:.1f}%")

    print(f"\nValidation Dataset:")
    print(f"  Total samples: {len(val_df)}")
    print(f"  Positive samples: {sum(val_df['label'])}")
    print(f"  Negative samples: {len(val_df) - sum(val_df['label'])}")
    print(f"  Positive ratio: {sum(val_df['label']) / len(val_df) * 100:.1f}%")

    print(f"\nTest Dataset:")
    print(f"  Total samples: {len(test_df)}")
    print(f"  Positive samples: {sum(test_df['label'])}")
    print(f"  Negative samples: {len(test_df) - sum(test_df['label'])}")
    print(f"  Positive ratio: {sum(test_df['label']) / len(test_df) * 100:.1f}%")

    # Display sample data
    print(f"\n" + "=" * 50)
    print("SAMPLE DATA")
    print("=" * 50)

    print(f"\nTraining Data Samples:")
    print(train_df.head(10).to_string(index=False))

    print(f"\nValidation Data Samples:")
    print(val_df.head(5).to_string(index=False))

    print(f"\nTest Data Samples:")
    print(test_df.head(5).to_string(index=False))

    print(f"\n" + "=" * 50)
    print("FILES CREATED")
    print("=" * 50)
    print(f"✅ data/train.csv - {len(train_df)} samples")
    print(f"✅ data/validation.csv - {len(val_df)} samples")
    print(f"✅ data/test.csv - {len(test_df)} samples")

    # Text length analysis
    print(f"\n" + "=" * 50)
    print("TEXT LENGTH ANALYSIS")
    print("=" * 50)

    all_texts = train_texts + val_texts + test_texts
    text_lengths = [len(text) for text in all_texts]

    print(f"Average text length: {np.mean(text_lengths):.1f} characters")
    print(f"Median text length: {np.median(text_lengths):.1f} characters")
    print(f"Min text length: {np.min(text_lengths)} characters")
    print(f"Max text length: {np.max(text_lengths)} characters")
    print(f"95th percentile: {np.percentile(text_lengths, 95):.1f} characters")

    return train_df, val_df, test_df


if __name__ == "__main__":
    print("🚀 Starting sample data generation for MLOps DistillBERT Pipeline")
    print("=" * 70)

    # Generate the data
    train_df, val_df, test_df = generate_sample_data()

    print("\n🎉 Sample data generation completed successfully!")