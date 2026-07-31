import os #import pythons built in os module 
        #used to communicate with pc for dataset stored folders and subfolders 
import splitfolders
DATA_DIR ="data/raw" # called a constant
OUTPUT_DIR = "data"
def check_dataset():
    if not os.path.exists(DATA_DIR):#checks whether data set is present or not !
        print("Dataset folder not found!")
        raise FileNotFoundError("Dataset folder is not found! ")
def get_classes():
    return os.listdir(DATA_DIR)
def count_images(classes):
    print("\n Image Count")
    print("-"*30)
    
    for cls in classes:
        class_path = os.path.join(DATA_DIR, cls)
        images = os.listdir(class_path)
        print(f"{cls}: {len(images)} images")
def split_dataset():
    #splitting data set into train,validation and test sets 
    if os.path.exists(os.path.join(OUTPUT_DIR,"train")):
        print("\n Data set is already split")
        return 
    print("\nSplitting data set")
    
    splitfolders.ratio(
        input= DATA_DIR,
        output = OUTPUT_DIR,
        seed = 42,
        ratio = (0.8, 0.1, 0.1)
    )
    print("Data split completed")

def main():
    check_dataset()
    classes = get_classes()
    print("=" *40)
    print("Potato Disease Dataset")
    print("=" * 40)

    print(f"\nTotal Classes : {len(classes)}\n")

    print("Class Names")
    print("-" * 30)

    for cls in classes:
        print(cls)

    count_images(classes)
    split_dataset()
if __name__ == "__main__":
    main()
