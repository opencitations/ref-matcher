# Process Citations using Grobid

This initial step of the project processes unstructured citations from a `.json.gz` file, extracts them, writes them to `.txt` files, and uses Grobid's `processCitationsList` service to convert the unstructured citations into structured XML files.

## Installation Guide

### Step 1: Clone the Grobid repository and run the service

1. First, install **Grobid** on your machine:
   ```bash
   git clone https://github.com/kermitt2/grobid.git
   cd grobid

2. Install Java and Maven (if not already installed):
    ```bash
    sudo apt update
    sudo apt install default-jdk maven

3. Build and start the Grobid service:

   ```bash
  
      ./gradlew clean install
      ./gradlew run

 By default, the service will run on http://localhost:8070.

### Step 2: Install the Grobid Python Client

 1. Clone the Grobid Python Client:

    ```bash

    git clone https://github.com/kermitt2/grobid-client-python.git
    cd grobid-client-python


### Step 3: Prepare your environment

 1. Make sure the Grobid service is running:

    ```bash

      ./gradlew run

Clone this repository and navigate to the project directory where the script structure_citations.py is located.

Ensure that the input file 0.json.gz is placed in the same directory as the script.

### Step 4: Running the script

1. Execute the Python script:

    ```bash

      python structure_citations.py

  The script will:
      Read the 0.json.gz file, extract unstructured citations, and write them to .txt files inside the unstructured/ folder.
      Call the Grobid service to convert the unstructured citations into structured XML files, saving them in the structured/ folder.

### Step 5: Check the output

The structured XML files will be saved in the structured/ directory. You can inspect them to verify that the citations have been correctly structured.

