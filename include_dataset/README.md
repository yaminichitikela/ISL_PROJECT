---
language:
- en
license: cc-by-4.0
size_categories:
- 1K<n<10K
pretty_name: INCLUDE
dataset_info:
  features:
  - name: parent_label
    dtype: string
  - name: label
    dtype: string
  - name: video_path
    dtype: string
  - name: include_50
    dtype: bool
  splits:
  - name: train
    num_bytes: 236952
    num_examples: 3816
  - name: val
    num_bytes: 26481
    num_examples: 425
  - name: test
    num_bytes: 62432
    num_examples: 1009
  download_size: 114570
  dataset_size: 325865
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
---

# Dataset Card for INCLUDE

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Summary](##dataset-summary)
- [Video download instructions](###video-download-instructions)
- [Supported Tasks](###supported-tasks)
- [Data Splits](###data-splits)
- [Dataset Creation](###dataset-creation)
- [Personal and Sensitive Information](###personal-and-sensitive-information)
- [Discussion of Biases](###discussion-of-biases)
- [Citation Information](###citation-information)
- [Contributions](###contributions)
 
### Dataset Summary

This dataset contains all videos in the INCLUDE dataset. As huggingface does not support video uploads at this time, the HF dataset contains metadata about each video such as the parent class, the video class, the path to the video and whether its a part of the INCLUDE-50 dataset (use include_50==True to get only include_50 videos). 
The videos themselves can be downloaded from [Zenodo](https://zenodo.org/records/4010759) using the provided bash script.  

### Video download instructions 

1. Copy paste the following code into a file. Save the file as `download_script.sh`
```
#!/bin/bash

# The base URL for the API request
base_url="https://zenodo.org/api/records/4010759"

# Fetch the JSON metadata from Zenodo
response=$(curl -s "$base_url")

# Parse JSON to extract file URLs and names using jq
echo "$response" | jq -r '.files[] | .links.self + " " + .key' | while read -r file_url file_name
do
  # Use curl to download each file and save it with the respective name
  echo "Downloading $file_name from $file_url..."
  curl -o "$file_name" "$file_url"
  echo "$file_name downloaded."
done

echo "All files downloaded."

# Loop through all zip files in the current directory
for file in *.zip; do
    # Unzip each file into a directory with the same name as the zip file without the extension
    unzip "${file%.zip}"
done

echo "All files unzipped."

```
2. Make the above file executable by opening a terminal window and running `chmod +x download_script.sh`
3. Execute the above file in the directory you want to store the videos by running `./download_script.sh`

Use the video_paths mentioned in the huggingface 'video_path' column to access the corresponding video. 

### Supported Tasks

This dataset was created for the goals of education and Isolated Sign Language Recognition, but may be used for other purposes.

### Data Splits

The dataset is split into train and test splits as described in the paper

### Dataset Creation

All details on dataset creation can be found in the INCLUDE paper. 

### Personal and Sensitive Information

These videos represent real people and their unique ways of communication. We urge all users of the dataset to respect their privacy - do not use these videos for any purposes that might infringe on the privacy or dignity of the individuals featured. Please ensure that the usage of these videos aligns with ethical standards and promotes understanding and inclusivity.

### Discussion of Biases

India is a diverse country, and does not have one uniform sign language. The videos in this dataset were shot in Chennai, Tamil Nadu. However, they are not the only representation of "Indian Sign Language", as ISL varies from place to place across the country.

### Citation Information

If you use this dataset, please cite the following work:


@inproceedings{sridhar_include:_2020,
	address = {New York, NY, USA},
	series = {{MM} '20},
	title = {{INCLUDE}: {A} {Large} {Scale} {Dataset} for {Indian} {Sign} {Language} {Recognition}},
	isbn = {9781450379885},
	shorttitle = {{INCLUDE}},
	url = {https://doi.org/10.1145/3394171.3413528},
	doi = {10.1145/3394171.3413528},
    urldate = {2024-07-16},
	booktitle = {Proceedings of the 28th {ACM} {International} {Conference} on {Multimedia}},
	publisher = {Association for Computing Machinery},
	author = {Sridhar, Advaith and Ganesan, Rohith Gandhi and Kumar, Pratyush and Khapra, Mitesh},
	month = oct,
	year = {2020},
	pages = {1366--1375},
}


### Contributions

Thanks to [@Rohith](https://github.com/grohith327), [@Gokul](https://huggingface.co/GokulNC) and [@Advaith](https://github.com/Ads-cmu/) for adding this dataset. For any further questions, please reach out to advaithsridhar08@gmail.com