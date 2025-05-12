Instructions to Execute the Source Code -

1. Run the command : pip install -r requirements.txt 
    This is done to install all dependencies and libraries used in the program

2. Run the real_time_detector.py using command : python real_time_detector.py 
   This file contains the UI structure as well as fetching details for loading the AI Model, and system utilities used
   Note: Depending on the system, this can take up to a couple minutes as some necessary files are initialised and psutil searches for 
         access to required information.

3. Under the UI there will be 5 options named - 
    a) Refresh
    b) Auto Refresh
    c) Detect Anomalies
    d) Save Snapshot
    e) Show Process Table
    As soon as the UI window pops up, click the 'Auto Refresh' button, and let it run to collect system info. The longer it runs, the 
    more info is collected. A minimum time of 1-2 minutes is suggested.

4.  Then press the 'Detect Anomaly' button.
    The system will pause for a bit to let the model run and then records the anomalous processes and stores them into a .json file 
    with the name formatted as 'anomaly_report_YYMMDD_HHMMSS.json' with system readings details about the anomalous process. The 
    number of anomalies depend on the native server it is working on as well as training time. This method can be used in smart data 
    centre for their server energy reduction and optimisation.

**Note:**

1) An example `.json` log is provided as `ExampleJSONLogFile.json`.

2) On some systems, a warning stating  
   *"oneDNN custom operations are on. You may see slightly different numerical results due to  
   floating-point round-off errors from different computation orders. To turn them off, set the environment variable  
   `TF_ENABLE_ONEDNN_OPTS=0`"*  
   is seen. This warning does not affect the program and is due to TensorFlow library version differences  
   on different systems.

