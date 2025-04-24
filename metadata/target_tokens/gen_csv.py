import pandas as pd
import os

def get_line_index(feature, category_file):
    """Find the line index (0-based) of a feature in a category file"""
    try:
        with open(category_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if feature.strip() == line.strip():
                    return i
        return -1  # Feature not found
    except FileNotFoundError:
        return -1

def determine_category(feature):
    """Determine which category file contains the feature"""
    category_files = {
        'articulation': '/home/yzhong/gits/interpretable-pd/metadata/disvoice/articulation.txt',
        'glottal': '/home/yzhong/gits/interpretable-pd/metadata/disvoice/glottal.txt',
        'phonation': '/home/yzhong/gits/interpretable-pd/metadata/disvoice/phonation.txt',
        'prosody': '/home/yzhong/gits/interpretable-pd/metadata/disvoice/prosody.txt'
    }
    
    for category, file_path in category_files.items():
        if get_line_index(feature, file_path) != -1:
            return category, file_path
    
    return None, None

def get_stat_id(feature):
    """Extract the stat_id from the feature name"""
    if feature.startswith('average '):
        return 'average'
    elif feature.startswith('std '):
        return 'std'
    elif feature.startswith('skewness '):
        return 'skewness'
    elif feature.startswith('kurtosis '):
        return 'kurtosis'
    elif feature.startswith('global average '):
        return 'average'
    elif feature.startswith('global std '):
        return 'std'
    else:
        return 'ratio'

# def extract_feature_id(feature, stat_id):
#     """Extract the base feature name from the full feature string"""
#     if stat_id == 'ratio':
#         return feature
#     elif feature.startswith('global '):
#         # Handle global features, which have a more complex structure
#         parts = feature.split(' ', 2)
#         if len(parts) >= 3:
#             return parts[2]
#         return feature
#     else:
#         # Regular features like "average Jitter", "std Shimmer", etc.
#         return feature[len(stat_id)+1:]

def main(txt_file_path):
    # Read VoiceQuality.txt
    with open(txt_file_path, 'r') as f:
        voice_quality_features = [line.strip() for line in f.readlines() if line.strip()]
    
    # Initialize data structures
    data = {
        'index_pos': [],
        'feature_id': [],
        'stat_id': [],
        'category_id': []
    }
    
    # Process each feature
    for feature in voice_quality_features:
        # Get the stat_id (average, std, etc.)
        stat_id = get_stat_id(feature)
        
        # Extract the actual feature ID
        feature_id = feature
        
        # Determine which category this feature belongs to
        category, category_file = determine_category(feature)
        
        if category:
            # Get the line index in the category file
            index_pos = get_line_index(feature, category_file)
            
            # Add to data
            data['index_pos'].append(index_pos)
            data['feature_id'].append(feature_id)
            data['stat_id'].append(stat_id)
            data['category_id'].append(category)
        else:
            # For features not found in any category file (might be composite features)
            data['index_pos'].append(-1)
            data['feature_id'].append(feature_id)
            data['stat_id'].append(stat_id)
            
            # Try to determine category based on feature name
            if 'GCI' in feature or 'NAQ' in feature or 'QOQ' in feature or 'H1H2' in feature or 'HRF' in feature:
                data['category_id'].append('glottal')
            elif 'Jitter' in feature or 'Shimmer' in feature or 'APQ' in feature or 'PPQ' in feature:
                data['category_id'].append('phonation')
            else:
                data['category_id'].append('unknown')
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    output_path = txt_file_path.replace('.txt', '.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} entries.")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])