from GetData import get_data
from Preprocessing import preprocess_data

def main():
    merged_df = get_data()

    print("Merged Data:")
    print(merged_df)

    processed_df = preprocess_data(merged_df)

    print("Processed Data:")
    print(processed_df)

if __name__ == '__main__':
    main()
