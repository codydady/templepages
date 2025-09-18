import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time
from collections import defaultdict
from functools import wraps
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import psutil

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Timing decorator
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"\nTIMING: {func.__name__} executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

# Load model
@timing_decorator
def load_model():
    print("🔮 Loading TensorFlow model...")
    model = tf.keras.models.load_model(os.path.expanduser("./idol_classifier.keras"))
    if tf.config.list_physical_devices('GPU'):
        print("⚡ GPU detected - enabling batch prediction")
        model.make_predict_function()
    return model

# Settings
img_size = (224, 224)
batch_size = 16  # Optimal for CPU-heavy workloads
class_names = ['ambal', 'hanuman', 'murugan', 'others', 'pillaiyar', 'shivan', 'temple']
model = load_model()

def is_prefixed(filename):
    """Check if filename already has a class prefix"""
    return any(filename.lower().startswith(f"{cls}_") for cls in class_names)

def process_image_batch(batch_args):
    """Process a batch of images with rename checking"""
    batch_paths, rename_files = batch_args
    results = []
    
    try:
        # Batch preprocessing
        batch_images = []
        valid_paths = []
        for img_path in batch_paths:
            if rename_files and is_prefixed(os.path.basename(img_path)):
                results.append(("skip", 1.0, os.path.basename(img_path), None))
                continue
                
            img = Image.open(img_path).convert('RGB').resize(img_size)
            batch_images.append(np.array(img)/255.0)
            valid_paths.append(img_path)
        
        # Batch prediction only for non-prefixed images
        if valid_paths:
            predictions = model.predict(np.array(batch_images), verbose=0)
            
            for i, img_path in enumerate(valid_paths):
                max_conf = round(float(np.max(predictions[i])), 2)
                predicted_class = class_names[np.argmax(predictions[i])]
                base_name = os.path.basename(img_path)
                
                if predicted_class != "temple" and rename_files:
                    new_name = f"{predicted_class}_{base_name}"
                    new_path = os.path.join(os.path.dirname(img_path), new_name)
                    os.rename(img_path, new_path)
                    results.append((predicted_class, max_conf, base_name, new_name))
                else:
                    results.append((predicted_class, max_conf, base_name, None))
    
    except Exception as e:
        print(f"\n⚠️ Batch error: {str(e)}")
        for img_path in batch_paths:
            results.append((None, 0, os.path.basename(img_path), None))
    
    return results

@timing_decorator
def predict_on_images_parallel(directory, rename_files=True):
    """Process images with batch optimization"""
    # Gather all image paths
    image_paths = []
    for root, _, files in os.walk(os.path.expanduser(directory)):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    total_images = len(image_paths)
    class_counts = defaultdict(int)
    skipped_files = 0
    print(f"🔍 Found {total_images:,} images to process")
    print(f"💻 Using {max(1, cpu_count()-1)} CPU cores")
    print(f"📦 Batch size: {batch_size} images")

    # Create batches
    batches = [
        (image_paths[i:i + batch_size], rename_files)
        for i in range(0, len(image_paths), batch_size)
    ]

    # Process in parallel
    with Pool(processes=max(1, cpu_count()-1)) as pool:
        results = []
        with tqdm(total=len(batches), unit='batch', desc="🚀 Processing", ncols=100) as pbar:
            for batch_results in pool.imap_unordered(process_image_batch, batches):
                results.extend(batch_results)
                pbar.update(1)
                pbar.set_postfix_str(f"CPU: {psutil.cpu_percent()}% | Images: {len(results):,}/{total_images:,}")

    # Process results
    successful = 0
    for predicted_class, max_conf, old_name, new_name in results:
        if predicted_class == "skip":
            skipped_files += 1
            continue
        if predicted_class:
            class_counts[predicted_class] += 1
            successful += 1
            if new_name:
                print(f"\r✅ Renamed: {old_name[:30]}... → {new_name[:30]}... (Confidence: {max_conf:.0%})", end='')

    # Summary
    print("\n\n📊 ========== RESULTS ==========")
    for class_name in class_names:
        print(f"{class_name.capitalize():<9}: {class_counts[class_name]:>6,}")
    
    print(f"\n🌐 Total images: {total_images:,}")
    print(f"✔️  Successful: {successful:,}")
    print(f"🔁 Skipped (already prefixed): {skipped_files:,}")
    print(f"❌ Failed: {total_images - successful - skipped_files:,}")

if __name__ == "__main__":
    print("🛠️ Starting image classification pipeline")
    start_time = time.perf_counter()
    
    target_directory = "/Users/sriram/Desktop/loft/idolsort"
    print(f"\n📂 Processing directory: {target_directory}")
    
    predict_on_images_parallel(target_directory)
    
    total_time = time.perf_counter() - start_time
    print(f"\n⏱️ TOTAL EXECUTION TIME: {total_time/60:.2f} minutes")
