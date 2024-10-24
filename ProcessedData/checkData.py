import pandas as pd 
import re

# Here setup the path to make Easy entry
text = './Dataset/IMDB Dataset.csv'

# Read the csv file to check the structure of the data 
imdb = pd.read_csv(text)

# Check for missing data 
missing = imdb.isna().sum()

# Check for inconsistent data
inconsistent = imdb.dtypes

# Check for duplicates data
duplicates = imdb.drop_duplicates()



def PreprocessData(df, text_cloumn):

    def preprocess_text(text):
        # Combining two patterns: one for HTML tags and one for special characters
        pattern = r'[^a-zA-Z0-9\s]|<[^>]+>'

        # Use re.sub() to remove the matched pattern (special characters and HTML tags)
        tagHtmlRemove = re.sub(pattern, " ", text)

        # Normalize whitespace (replace multiple spaces with a single space)
        tagHtmlRemove = re.sub(r'\s+', ' ', tagHtmlRemove).strip()

        print(f"The processed text contains no special characters: '{tagHtmlRemove}'")

        # Convert to lowercase
        lowercase = tagHtmlRemove.lower()

        return lowercase

    df["Processed text"] = df[text_cloumn].apply(preprocess_text)

    return df

def saveText(df, filename="processed_imdb_reviews.csv"):
    df.to_csv(filename, index=False)
    print(f"Processed IMDB datasaet saved to {filename}")


processedData = PreprocessData(imdb, 'review')

saveText(processedData)


print(processedData.head())

