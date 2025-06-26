#################################################################################################################
#################################################################################################################
#################################################################################################################

import os  # Import OS module for file and directory operations
from tkinter import *  # Import all functions from Tkinter for GUI
import numpy as np  # Import NumPy for numerical operations
from PIL import Image, ImageTk  # Import PIL for image processing in Tkinter
import tensorflow as tf  # Import TensorFlow for deep learning models
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator for image augmentation
from tensorflow.keras.models import Sequential  # Import Sequential model for building neural networks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # Import layers for CNN model
import re  # Import re module for regular expressions
import ast  # Import ast module for parsing Python code as syntax trees
from collections import deque  # Import deque from collections for efficient queue operations

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

Debug = False # Debug bool that lowers eta time and resets all locations

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

TrainingData = "D:\Python Codes\Fire\FireClass"  # Path to training dataset

ImgWidth, ImgHeight = 150, 150  # Image dimensions
batch_size = 32  # Batch size for training
NoClasses = 13  # Number of classes in dataset

DataGenerator = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

TrainGenerator = DataGenerator.flow_from_directory(  # Load and preprocess images
    TrainingData, 
    target_size=(ImgWidth, ImgHeight), 
    batch_size=batch_size, 
    class_mode='categorical'
)

class_labels = sorted(os.listdir(TrainingData))  # Get sorted class labels

model = Sequential([  # Define CNN model
    Conv2D(32, (3, 3), activation='relu', input_shape=(ImgWidth, ImgHeight, 3)),  # First convolution layer
    MaxPooling2D((2, 2)),  # First pooling layer
    
    Conv2D(64, (3, 3), activation='relu'),  # Second convolution layer
    MaxPooling2D((2, 2)),  # Second pooling layer
    
    Conv2D(128, (3, 3), activation='relu'),  # Third convolution layer                                                          
    MaxPooling2D((2, 2)),  # Third pooling layer
    
    Conv2D(128, (3, 3), activation='relu'),  # Fourth convolution layer
    MaxPooling2D((2, 2)),  # Fourth pooling layer
    
    Flatten(),  # Flatten layer for fully connected input
    Dense(512, activation='relu'),  # Dense layer with 512 neurons
    Dense(NoClasses, activation='softmax')  # Output layer with softmax activation
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile model

model.fit(TrainGenerator, epochs=10)  # Train model for 10 epochs

#################################################################################################################

def ImageAI(ImageFile):  # Function to predict class of an image
    CurrentImage = tf.keras.preprocessing.image.load_img(ImageFile, target_size=(ImgWidth, ImgHeight))  # Load image and resize
    ProcessingImage = tf.keras.preprocessing.image.img_to_array(CurrentImage)  # Convert image to array
    ProcessingImage = np.expand_dims(ProcessingImage, axis=0) / 255.0  # Expand dimensions and normalize
    
    ImagePrediction = model.predict(ProcessingImage)  # Get model prediction
    Classes = np.argmax(ImagePrediction)  # Get class index with highest probability
    ImageClass = class_labels[Classes]  # Get class label
    
    return ImageClass  # Return predicted class

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def BoroughOuput(FilePath, CurrentBorough):  # Function to process station data for a borough
    global FullAdress  # Ensure FullAdress is accessible if needed
    global NotHome  # Track units not at their station
    
    with open(FilePath, 'r') as File:  # Open the file and read contents
        FileContents = File.read().strip()  # Read and remove leading/trailing whitespace

    # Split contents into separate stations based on double line breaks
    Stations = re.split(r'\n\s*\n', FileContents)
    Output = []  # List to store formatted station details
    NotHome = 0  # Counter for units not at their station

    for Station in Stations:
        Lines = Station.splitlines()  # Split station data into lines
        if len(Lines) < 4:  # Skip if there aren't enough details
            continue
        
        StationBorough = Lines[0].strip()  # Extract borough name
        if StationBorough.lower() != CurrentBorough.lower():  # Skip stations outside the requested borough
            continue
        
        StationName = Lines[1].strip()  # Extract station name
        StationAddress = Lines[2].strip()  # Extract station address
        
        # Extract the unit code and address pairs from the fourth line
        LineSection = Lines[3].strip()
        try:
            CurrentLine = ast.literal_eval(LineSection)  # Convert string to a Python list
        except:
            pass  # Ignore errors in parsing
        
        OutputContent = []  # List to store unit status at the station
        for Entry in CurrentLine:
            if len(Entry) < 2:  # Skip invalid entries
                continue
            UnitCode = Entry[0].strip()  # Extract unit code
            UnitAddress = Entry[1].strip()  # Extract unit's reported address
            
            # If the unit is at the station, mark it as "at Station"
            if UnitAddress == StationAddress:
                OutputContent.append(f"{UnitCode} at Station")
            else:
                # If the unit is "on route," keep that status
                if "on route" in UnitAddress.lower():
                    OutputContent.append(f"{UnitCode} {UnitAddress}")
                else:
                    OutputContent.append(f"{UnitCode} at {UnitAddress}")  # Otherwise, show location
                NotHome += 1  # Increment counter for units not at station
        
        if OutputContent:  # If there is relevant data, format it
            OutputLine = StationName + "\n" + "\n".join(OutputContent)
            Output.append(OutputLine)
    
    FinalOutput = "\n\n".join(Output)  # Join all stations' data into one formatted output
    return FinalOutput  # Return the formatted station details

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def Clear():  # Function to reset station data to default
    DefultStations = open("DefultStations.txt", "r")  # Open the default station file for reading
    ResetStations = DefultStations.read()  # Read the default station data
    DefultStations.close()  # Close the file

    # Overwrite the current station file with the default data
    CurrentStations = open("Stations.txt", "w")  # Open the current station file for writing
    CurrentStations.write(ResetStations)  # Write the default data into it
    CurrentStations.close()  # Close the file   

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def Logic():  # Function to determine fire dispatch units and estimated response times
    Dispatch = GetPDA(IncidentType, ImageClass)  # Get Pre-Determined Attendance (PDA) based on incident type and AI classification
    CurrentData = []  # List to store formatted dispatch details
    UnitsNeeded = []  # List to track required unit types and quantities

    # Since 'global' is used in GetPDA, these variables are accessible here:
    if P > 0:
        UnitsNeeded.append(("P", P))  # Pump Ladder
    if PL > 0:
        UnitsNeeded.append(("PL", PL))  # Pump
    if FRU > 0:
        UnitsNeeded.append(("FRU", FRU))  # Fire Rescue Unit
    if Fireboat > 0:
        UnitsNeeded.append(("Fireboat", Fireboat))  # Fireboat
    if CU > 0:
        UnitsNeeded.append(("CU", CU))  # Command Unit
    if ALP > 0:
        UnitsNeeded.append(("ALP", ALP))  # Aerial Ladder Platform
    # Add more unit codes as needed

    # Retrieve available units based on location and station data
    CurrentUnit, Null = GetUnits(HalfPostcode, StationInfo, Graph, UnitsNeeded)  
    # CurrentUnit => { "P": [ (station, qty, BFSdist), ...], "PL": [...], ...}

    # Build response details with estimated time of arrival (ETA) for each unit
    if CurrentUnit:
        for TypeNeeded, CurrentDispatch in CurrentUnit.items():  # Iterate through unit types
            if CurrentDispatch is None:
                pass  # Skip if no dispatch info
            else:
                for (CurrentStation, UnitNo, Distance) in CurrentDispatch:  # Process unit dispatch details
                    if Distance <= 1:  # If distance is 1km or less, assume quick response
                        ETA = 5
                    else:
                        ETA = Distance * 5  # Estimate ETA (5 minutes per km)
                    CurrentData.append(f"{UnitNo} {TypeNeeded} from {CurrentStation['Station']} (ETA {ETA} minutes).")  # Format output

    # Format output into multiple lines if needed
    PerLine = 4  # Adjust if required for readability
    Lines = []
    for i in range(0, len(CurrentData), PerLine):  # Split into rows
        RowOutput = CurrentData[i:i+PerLine]
        Lines.append("  ".join(RowOutput))
    
    CurrentDataString = "\n".join(Lines)  # Join formatted dispatch details
    CurrentDataString = (f"\n...\nRosey says: There's a fire at {Adress} {Borough} {Postcode} I recommend:\n{CurrentDataString} {PDA}").strip()

    # Display the dispatch recommendation in the GUI
    RoseyText.configure(state='normal')  
    RoseyText.insert("end", CurrentDataString)
    RoseyText.configure(state='disable')

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################
    
def DisplayInfo(x):  # Function to display fire station info for a selected borough
    global CurrentBorough  
    InfoBox.configure(state='normal')  # Enable text box for editing
    InfoBox.delete('1.0', END)  # Clear previous content
    InfoBox.configure(state='disabled')  # Disable editing again

    FilePath = 'Stations.txt'  # Path to the file containing station data  
    CurrentBorough = x  # Set the selected borough for filtering station data
    Output = BoroughOuput(FilePath, CurrentBorough)  # Get station details for the borough

    # Display the retrieved information in the InfoBox
    InfoBox.configure(state='normal')  # Enable text box for inserting data
    InfoBox.insert("end", "".join(Output))  # Insert formatted station info
    InfoBox.configure(state='disable')  # Disable editing again to prevent modifications

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def UpdateButtons():  # Function to update button colors based on unit availability
    
    FilePath = 'Stations.txt'  # Path to the file containing station data
        
    for StationBorough in AllButtons:
        # Call BoroughOuput to retrieve unit data for this borough.
        # This function updates the global variable NotHome.
        Null = BoroughOuput(FilePath, StationBorough)
        
        # Change button color based on the number of units not at their station
        if NotHome >= 5:
            AllButtons[StationBorough].config(bg="red")  # High alert - Many units away
        elif NotHome >= 3:
            AllButtons[StationBorough].config(bg="orange")  # Moderate alert
        elif NotHome >= 1:
            AllButtons[StationBorough].config(bg="yellow")  # Low alert - Some units away
        else:
            AllButtons[StationBorough].config(bg="spring green")  # All units at station - Normal status

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################
            
def Update():  # Function to process and update fire dispatch information
    global NextFile
    global Adress, Borough, Postcode, IncidentType, LiveAtRisk, ImageClass
    global P, PL, FRU, Fireboat, CU, ALP
    global HalfPostcode, StationInfo, Graph  

    try:  
        # Read the latest call transcript
        Transcript = open("CallTranscript"+str(NextFile)+".txt", "r")  
        
        # Clear InfoBox display before updating
        InfoBox.configure(state='normal')  
        InfoBox.delete('1.0', END)  
        InfoBox.configure(state='disabled')  

        # Extract relevant details from the transcript
        CallData = Transcript.readlines()  
        Adress = CallData[1].strip()  
        Borough = CallData[2].strip()  
        Postcode = CallData[3].strip()  
        IncidentType = CallData[6].strip()  
        LiveAtRisk = CallData[9].strip()  

        ##################################################################################################
        
        def GetDefult(FileName):  # Function to retrieve default station addresses
            with open(FileName, 'r', encoding='utf-8') as File:  
                FileContents = File.read().strip()  
            
            Stations = re.split(r'\n\s*\n', FileContents)  # Split data into individual stations
            DefultInfo = {}  # Dictionary to store default station data
            
            for Row in Stations:  
                Lines = Row.splitlines()  
                if len(Lines) < 3:  # Skip incomplete entries  
                    continue  

                StationName = Lines[1].strip()  # Extract station name  
                StationAddress = Lines[2].strip()  # Extract station address  
                DefultInfo[StationName] = StationAddress  # Store in dictionary  
            
            return DefultInfo  

        ##################################################################################################

        # Load station network graph and current station details
        MapData = "Adjancent.txt"  
        Graph = GraphAdjacency(MapData)  
        StationFile = "Stations.txt"  
        StationInfo = SplitStations(StationFile)  
        HalfPostcode = GetHalf(Postcode)  

        ##################################################################################################

        if IncidentType == "Call Back":  # Handling returning units to station
            FullAdress = Adress  
            StationName = CallData[1].strip()  # Extract station name from transcript  
            UnitCode = CallData[9].strip()  # Extract unit type  

            CurrentRecord = None  
            for NextStation in StationInfo:  
                if NextStation['Station'] == StationName:  
                    CurrentRecord = NextStation  
                    break  

            if CurrentRecord is None:  # If station is not found, notify and exit
                RoseyText.configure(state='normal')  
                RoseyText.insert("end", f"\n...\nRosey says: No current Station found for Station '{StationName}'.")  
                RoseyText.configure(state='disabled')  
                NextFile += 1  
                return  

            # Retrieve unit's current location  
            if 'CodeAddress' in CurrentRecord:  
                CurrentLocation = CurrentRecord['CodeAddress'].get(UnitCode, CurrentRecord['StationAddress'])  
            else:  
                CurrentLocation = CurrentRecord['StationAddress']  # Default to station address  

            # Get the default station address for the unit's home location
            DefultInfo = GetDefult("DefultStations.txt")  
            HomeAddress = DefultInfo.get(StationName, CurrentRecord['StationAddress'])  

            # Extract postcode sections for distance calculation
            StationHalf = ReturnHalf(CurrentLocation)  
            HomeHalf = ReturnHalf(HomeAddress)  
            if not StationHalf:  
                StationHalf = CurrentRecord['StationAddress'].split()[-1]  
            if not HomeHalf:  
                HomeHalf = HomeAddress.split()[-1]  

            # Calculate travel distance and estimate arrival time
            Distance = GetDistance(Graph, StationHalf, HomeHalf)  
            if Distance is None:  
                Distance = 2  # Fallback distance  
            ETAMins = 5 if Distance <= 1 else Distance * 5  # Estimate ETA in minutes  
            ETAMili = ETAMins * 60000  # Convert to milliseconds  

            # Schedule the unit's return update
            DelayedUpdate = [(StationName, UnitCode, None, ETAMili)]  
            ScheduledUpdates(DelayedUpdate, HomeAddress)  

            # Display return message
            RoseyText.configure(state='normal')  
            DispatchOutput = (  
                f"\n...\nRosey says: Returning unit {UnitCode} from current location {CurrentLocation} to {Adress} (ETA {ETAMins} minutes).\n"  
            )  
            RoseyText.insert("end", DispatchOutput)  
            RoseyText.configure(state='disabled')  

        else:  # Standard incident handling  
            ImageFile = r"D:\Python Codes\Fire\Image"+str(NextFile)+".png"  # Get the corresponding image  
            ImageClass = ImageAI(ImageFile)  # Classify fire type  
            Logic()  # Call logic function to determine dispatch  

        ##################################################################################################
        
        NextFile += 1  # Increment file counter for next update  

    except:  
        pass  # Prevents the program from crashing if an error occurs  

    DisplayInfo(CurrentBorough)  # Refresh borough display  
    UpdateButtons()  # Update UI buttons  
    root.after(1000, Update)  # Schedule next update in 1 second

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def GraphAdjacency(AdjancencyPath):  # Function to build a graph representation from adjacency list
    Graph = {}  # Dictionary to store graph structure
    
    with open(AdjancencyPath, 'r') as File:  # Open adjacency file
        for line in File:  
            line = line.strip()  # Remove leading/trailing whitespace
            if not line or ":" not in line:  # Skip empty lines or invalid formats
                continue  
            
            Left, Right = line.split(":", 1)  # Split into node and adjacent nodes
            Node = Left.strip()  # Clean node name
            Right = Right.replace('.', ',')  # Replace periods with commas for consistent separation
            
            # Extract and clean adjacent node names
            Adjancent = [n.strip() for n in Right.split(",") if n.strip()]  

            # Add node to graph if not already present
            if Node not in Graph:  
                Graph[Node] = []  
            
            # Add adjacent nodes to the node's adjacency list
            Graph[Node].extend(Adjancent)  
            
            # Ensure bidirectional connectivity by adding missing nodes
            for Neighbor in Adjancent:  
                if Neighbor not in Graph:  
                    Graph[Neighbor] = []  
    
    return Graph  # Return the constructed adjacency graph

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def GetDistance(Graph, Start, End):  # Function to find shortest path distance using BFS
    if Start not in Graph or End not in Graph:  # Check if both nodes exist in the graph
        return None  

    Visited = set()  # Set to keep track of visited nodes
    Queue = deque([(Start, 0)])  # Initialize queue with start node and distance 0
    Visited.add(Start)  # Mark start node as visited

    while Queue:  
        Current, Distance = Queue.popleft()  # Dequeue the next node and its distance
        
        if Current == End:  # If destination is reached, return distance
            return Distance  

        for Neighbor in Graph[Current]:  # Explore adjacent nodes
            if Neighbor not in Visited:  # If node hasn't been visited yet
                Visited.add(Neighbor)  # Mark as visited
                Queue.append((Neighbor, Distance + 1))  # Enqueue with incremented distance
    
    return None  # Return None if no path is found

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def GetHalf(FullPostcode):  # Function to extract the first part of a postcode
    Parts = FullPostcode.split()  # Split postcode into parts
    return Parts[0] if Parts else FullPostcode  # Return first part if available, else return original postcode

def ReturnHalf(FullPostcode):  # Function to extract the first part containing numbers from a postcode
    Tokens = FullPostcode.split()  # Split postcode into separate tokens
    for t in Tokens:  
        if 2 <= len(t) <= 4 and any(c.isdigit() for c in t):  # Check if token length is 2-4 and contains a digit
            return t  # Return the first valid part
    return None  # Return None if no valid part is found

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def SplitStations(FileName):  # Function to parse station data from a file
    with open(FileName, 'r', encoding='utf-8') as File:  
        FileContents = File.read().strip()  # Read file and remove extra spaces
    
    Stations = re.split(r'\n\s*\n', FileContents)  # Split data into individual station entries
    StationInfo = []  # List to store processed station data
    
    for Row in Stations:  
        Lines = Row.splitlines()  # Split each station's data into lines
        if len(Lines) < 4:  # Skip entries with missing data  
            continue  

        StationBorough = Lines[0].strip()  # Extract borough name  
        StationName = Lines[1].strip()  # Extract station name  
        StationAddress = Lines[2].strip()  # Extract station address  
        CodeLine = Lines[3].strip()  # Extract unit data (unit codes & addresses)  

        try:  
            Pairs = ast.literal_eval(CodeLine)  # Convert unit data string to list  
        except:  
            pass  # Skip if conversion fails  

        UnitNo = {}  # Dictionary to store unit counts  
        UnitLocation = None  # Track the location of the first unit  

        for i, Pair in enumerate(Pairs):  # Iterate through unit assignments  
            UnitCode = Pair[0].strip()  # Extract unit type (e.g., "P", "FRU")  
            UnitAddress = Pair[1].strip()  # Extract unit's assigned address  
            UnitNo[UnitCode] = UnitNo.get(UnitCode, 0) + 1  # Count occurrences of each unit type  

            if i == 0:  # Assign the location of the first listed unit  
                UnitLocation = UnitAddress  

        # Store extracted station information in a structured format  
        StationInfo.append({  
            'StationBorough': StationBorough,  # Borough name  
            'Station': StationName,  # Fire station name  
            'StationAddress': StationAddress,  # Fire station address  
            'EngineAdress': UnitLocation,  # Address of the first assigned unit  
            'UnitNo': UnitNo  # Dictionary of unit counts  
        })  

    return StationInfo  # Return list of parsed station data  

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def GetUnits(HalfPostcode, StationInfo, Graph, Requirements):  
    global FullAdress  # Ensure full address is accessible globally  
    StationData = []  # List to store station details and distances  

    # Process each station to determine distance and unit availability  
    for Station in StationInfo:  
        StationHalf = ReturnHalf(Station['EngineAdress'])  # Extract half postcode for station  
        Distance = GetDistance(Graph, StationHalf, HalfPostcode)  # Compute shortest distance  
        
        # Store station info along with distance and unit availability  
        StationData.append({  
            'Station': Station,  
            'Distance': Distance,  
            'UnitNo': dict(Station['UnitNo'])  # Copy unit counts to avoid modifying original data  
        })  

    # Sort stations by distance (handling None distances by placing them last)  
    StationData.sort(key=lambda x: x['Distance'] if x['Distance'] is not None else 9999)  

    CurrentUnit = {}  # Dictionary to store allocated units  
    DelayedUpdate = []  # List to store units needing delayed updates  

    # Process each required unit type and allocate from the nearest stations  
    for (TypeNeeded, QuantityNeeded) in Requirements:  
        AmountNeeded = QuantityNeeded  
        CurrentDispatch = []  # List to store dispatched units  

        for Info in StationData:  
            if Info['Distance'] is None:  # Skip stations without a valid distance  
                continue  
            Got = Info['UnitNo'].get(TypeNeeded, 0)  # Check available units of the required type  
            if Got <= 0:  # If no units are available, move to the next station  
                continue  
            
            Take = min(Got, AmountNeeded)  # Determine how many units to take  
            Info['UnitNo'][TypeNeeded] -= Take  # Deduct assigned units from station  
            AmountNeeded -= Take  # Reduce the remaining required amount  
            CurrentDispatch.append((Info['Station'], Take, Info['Distance']))  

            # Compute Estimated Time of Arrival (ETA)  
            if Info['Distance'] <= 1:  
                ETA = 5  # Minimum ETA  
            else:  
                ETA = Info['Distance'] * 5  # Assume 5 minutes per unit distance  
            
            ETA_time = ETA * 60000  # Convert ETA to milliseconds  
            if Debug == True:  # If debug mode is enabled, reduce ETA for faster testing  
                ETA_time = 10000  

            # Store the full incident address  
            FullAdress = f"{Adress}, {Borough}, {Postcode}"  

            # Schedule unit return updates  
            DelayedUpdate.append((Info['Station']['Station'], TypeNeeded, None, ETA_time))  

            cost_here = ETA  # Store cost based on computed ETA  

            if AmountNeeded <= 0:  # If all required units are assigned, stop searching  
                break  

        # If not enough units were found, mark as None  
        if AmountNeeded > 0:  
            CurrentUnit[TypeNeeded] = None  
        else:  
            CurrentUnit[TypeNeeded] = CurrentDispatch  

    # Schedule updates for dispatched units  
    ScheduledUpdates(DelayedUpdate, FullAdress)  
    return CurrentUnit, cost_here  # Return assigned units and estimated cost (ETA)  

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def ScheduledUpdates(DelayedUpdate, FinalAddress):  
    def ScheduledUpdate(StationName, UnitCode, FinalAddress, UpdatedText):     
        # Update the station file to reflect the unit's new location  
        UpdateFile("Stations.txt", [(StationName, UnitCode, FinalAddress)])  
        
        # Display the arrival message in the text widget  
        RoseyText.configure(state='normal')  
        RoseyText.insert("end", UpdatedText)  
        RoseyText.configure(state='disabled')  

    # Process each delayed update entry  
    for (StationName, UnitCode, _, ETA_time) in DelayedUpdate:  
        # Generate text for when the unit arrives  
        UpdatedText = f"\n...\nRosey says: {StationName} {UnitCode} has arrived at {FinalAddress}."  
        
        # Immediately mark the unit as "on route" in the station file  
        UpdateFile("Stations.txt", [(StationName, UnitCode, f"on route to {FinalAddress}")])  
        
        # Schedule the final arrival update after the estimated travel time  
        root.after(ETA_time, lambda SN=StationName, UC=UnitCode, FA=FinalAddress, UT=UpdatedText: ScheduledUpdate(SN, UC, FA, UT))  

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################
        
def UpdateFile(FileName, DispatchedUnits):  
    try:  
        # Read the existing station file
        with open(FileName, 'r', encoding='utf-8') as File:  
            FileContents = File.read().strip()  

        # Split file contents into separate station records  
        Stations = re.split(r'\n\s*\n', FileContents)  

        UpdatedInfo = []  # List to store updated station data  

        for Station in Stations:  
            Lines = Station.splitlines()  # Split each station entry into lines  
            if len(Lines) < 4:  # Skip entries that don't have enough details  
                UpdatedInfo.append(Station)  
                continue  

            # Extract station details  
            StationBorough = Lines[0].strip()  
            StationName = Lines[1].strip()  
            StationAddress = Lines[2].strip()  
            CodeLine = Lines[3].strip()  # Retrieve unit code-address pairs  

            try:  
                CurrentLine = ast.literal_eval(CodeLine)  # Convert string to list  
            except:  
                UpdatedInfo.append(Station)  # If conversion fails, keep original entry  
                continue  

            # Process dispatched units and update their location  
            for (DispatchedStation, UnitCode, NewAddress) in DispatchedUnits:  
                if StationName == DispatchedStation:  # Match station name  
                    for i, Entry in enumerate(CurrentLine):  
                        if Entry[0] == UnitCode:  # Find matching unit  
                            CurrentLine[i] = [UnitCode, NewAddress]  # Update unit location  

            # Reformat updated station record  
            UpdatedLine = str(CurrentLine)  
            UpdatedRecord = f"{StationBorough}\n{StationName}\n{StationAddress}\n{UpdatedLine}"  
            UpdatedInfo.append(UpdatedRecord)  

        # Write the updated station records back to the file  
        with open(FileName, 'w', encoding='utf-8') as File:  
            File.write("\n\n".join(UpdatedInfo))  

    except:  
        pass  # Prevents program crash if an error occurs  

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def GetPDA(x, y):
    # Fire Incidents
    global P, PL, FRU, Fireboat, CU, ALP, PDA
    if str(x) == "House Fire" and str(LiveAtRisk) == "No":
        PDA = "and 1 officer in charge"
        P = y
        PL = 1
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
        
    elif str(x) == "House Fire" and str(LiveAtRisk) == "Yes":
        PDA = "and 1 officer in charge, ambulance notified"
        P = y
        PL = 1
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "High-Rise Fire":
        PDA = "and multiple officers"
        P = y
        PL = 1
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Large Vehicle Fire":
        PDA = "and 1 officer in charge"
        P = 2
        PL = 1
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Bin/Small Rubbish Fire":
        PDA = ""
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Cladding Fire (High-Rise)":
        PDA = "and multiple officers"
        P = y
        PL = 3
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    elif str(x) == "Fire with Fatality":
        PDA = ", 2 FIU and press officer notified"
        P = y
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Fire Survival Guidance (FSG) Calls":
        PDA = "and personnel dispatched for complex life-risk situations"
        P = 0
        PL = 0
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    
    # Special Services (Rescues & Non-Fire Emergencies)
    elif str(x) == "Person Trapped (Non-Road Traffic Collision)":
        PDA = "and 1 officer in charge"
        P = 2
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 1
        ALP = 0
    elif str(x) == "Road Traffic Collision (Person Trapped)":
        PDA = "1 officer in charge, ambulance notified"
        P = 2
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 1
        ALP = 0
    elif str(x) == "Person Shut in Lift (Non-Emergency)":
        PDA = ""
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Animal Rescue (Large Animal)":
        PDA = "and specialist officers"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 1
        ALP = 0
    elif str(x) == "Flooding (Non-Commercial)":
        PDA = ""
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    
    # Hazardous Materials (HazMat) Incidents
    elif str(x) == "Gas Leak (Domestic/Commercial)":
        PDA = "and gas authority notified"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Minor Hazardous Spill (under 100L)":
        PDA = ""
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Major Hazardous Spill (over 100L)":
        PDA = "and 1 HazMat unit"
        P = 2
        PL = 0
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    elif str(x) == "Suspicious Substance/White Powder":
        PDA = ", 1 HazMat officer and police notified"
        P = 0
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Deliberate HazMat Release":
        PDA = ", HazMat and specialist units"
        P = 6
        PL = 0
        Fireboat = 0
        CU = 2
        FRU = 0
        ALP = 0
    
    # Other Emergencies
    elif str(x) == "Person Threatening to Jump/Rescue from Height":
        PDA = ", ambulance and police notified"
        P = 1
        PL = 1
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Person Collapsed Behind Locked Door":
        PDA = "and ambulance notified"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Vehicle into Building":
        PDA = ", 1 structural officer and ambulance"
        P = 2
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 1
        ALP = 0
    elif str(x) == "Train or Tram Crash":
        PDA = "and 1 USAR team"
        P = 4
        PL = 0
        Fireboat = 0
        CU = 1
        FRU = 1
        ALP = 0
    elif str(x) == "Aircraft Accident":
        PDA = "and specialist officers"
        P = y
        PL = 0
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    elif str(x) == "Fire on a Vessel":
        PDA = ""
        P = 3
        PL = 0
        Fireboat = 1
        CU = 1
        FRU = 0
        ALP = 0
        
    
    # Additional Response Triggers
    elif str(x) == "Multiple Calls (4 or more)":
        PDA = "and station commander dispatched"
        P = 0
        PL = 0
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    elif str(x) == "High-Rise Cladding Fire":
        PDA = "and senior officers"
        P = 8
        PL = 2
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    elif str(x) == "Confirmed Life Risk":
        PDA = "and ambulance dispatched immediately"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Large Fires (6+ P)":
        PDA = ", fire safety officers and senior officers dispatched"
        P = 0
        PL = 0
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    else:
        pass

    P = int(P)
    return PDA
    return P
    return PL
    return Fireboat
    return CU
    return FRU
    return ALP

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

root = Tk()                                                                                      #Tkinter Start

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

InfoBox = Text(root, state='disabled', height=65, width=45,  bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
InfoBox.place(x=1565, y=0)
                                                                                                   #Text Boxes
RoseyText = Text(root, state='disabled', height=7, width=180, bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
RoseyText.place(x=127, y=953)

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

LFBMap = PhotoImage(file="LFBMap.png")
RoseyIcon = PhotoImage(file="RoseyIcon.png")                                                    

Map = Label(root, image=LFBMap, bd=0)
Map.place(x=0, y=0)                                                                               #Needed Images
                                                                    
RoseyFace = Label(root, image=RoseyIcon, bd=0)
RoseyFace.place(x=0, y=953)

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

BtnBarkingAndDagenham = Button(root, text="Barking and\nDagenham", command=lambda: DisplayInfo("Barking and Dagenham"))
BtnBarnet= Button(root, text="Barnet", command=lambda: DisplayInfo("Barnet"))
BtnBexley= Button(root, text="Bexley", command=lambda: DisplayInfo("Bexley"))
BtnBrent = Button(root, text="Brent", command=lambda: DisplayInfo("Brent"))
BtnBromley= Button(root, text="Bromley", command=lambda: DisplayInfo("Bromley"))
BtnCamden = Button(root, text="Camden", command=lambda: DisplayInfo("Camden"))
BtnCity  = Button(root, text="City", command=lambda: DisplayInfo("City"))
BtnCroydon = Button(root, text="Croydon", command=lambda: DisplayInfo("Croydon"))
BtnEaling = Button(root, text="Ealing", command=lambda: DisplayInfo("Ealing"))
BtnEnfield = Button(root, text="Enfield", command=lambda: DisplayInfo("Enfield"))
BtnGreenwich = Button(root, text="Greenwich", command=lambda: DisplayInfo("Greenwich"))
BtnHackney = Button(root, text="Hackney", command=lambda: DisplayInfo("Hackney"))
BtnHammersmithAndFulham = Button(root, text="Hammersmith\nand Fulham", command=lambda: DisplayInfo("Hammersmith and Fulham"))
BtnHaringey = Button(root, text="Haringey", command=lambda: DisplayInfo("Haringey"))
BtnHarrow  = Button(root, text="Harrow", command=lambda: DisplayInfo("Harrow"))
BtnHavering  = Button(root, text="Havering", command=lambda: DisplayInfo("Havering"))
BtnHillingdon = Button(root, text="Hillingdon", command=lambda: DisplayInfo("Hillingdon"))
BtnHounslow = Button(root, text="Hounslow", command=lambda: DisplayInfo("Hounslow"))
BtnIslington = Button(root, text="Islington", command=lambda: DisplayInfo("Islington"))
BtnKensingtonAndChelsea= Button(root, text="Kensington\nand Chelsea", command=lambda: DisplayInfo("Kensington and Chelsea"))
BtnKingstonUponThames= Button(root, text="Kingston\nupon\nThames", command=lambda: DisplayInfo("Kingston upon Thames"))
BtnLambeth   = Button(root, text="Lambeth", command=lambda: DisplayInfo("Lambeth"))
BtnLewisham = Button(root, text="Lewisham", command=lambda: DisplayInfo("Lewisham"))
BtnMerton = Button(root, text="Merton", command=lambda: DisplayInfo("Merton"))
BtnNewham = Button(root, text="Newham", command=lambda: DisplayInfo("Newham"))
BtnRedbridge = Button(root, text="Redbridge", command=lambda: DisplayInfo("Redbridge"))
BtnRichmondUponThames  = Button(root, text="Richmond\nupon Thames", command=lambda: DisplayInfo("Richmond upon Thames"))
BtnSouthwark = Button(root, text="Southwark", command=lambda: DisplayInfo("Southwark"))
BtnSutton    = Button(root, text="Sutton", command=lambda: DisplayInfo("Sutton"))
BtnTowerHamlets = Button(root, text="Tower\nHamlets", command=lambda: DisplayInfo("Tower Hamlets"))
BtnWalthamForest= Button(root, text="Waltham\nForest", command=lambda: DisplayInfo("Waltham Forest"))
BtnWandsworth= Button(root, text="Wandsworth", command=lambda: DisplayInfo("Wandsworth"))
BtnWestminster = Button(root, text="Westminster", command=lambda: DisplayInfo("Westminster"))
BtnBarkingAndDagenham.place(x=1060, y=360)
BtnBarnet.place(x=620, y=200)
BtnBexley.place(x=1095, y=530)
BtnBrent.place(x=550, y=325)
BtnBromley.place(x=970, y=725)
BtnCamden.place(x=669, y=335)
BtnCity.place(x=782, y=414)
BtnCroydon.place(x=800, y=750)
BtnEaling.place(x=470, y=405)
BtnEnfield.place(x=775, y=125)
BtnGreenwich.place(x=957, y=505)
BtnHackney.place(x=825, y=332)
BtnHammersmithAndFulham.place(x=610, y=470)
BtnHaringey.place(x=750, y=255)
BtnHarrow.place(x=450, y=235)
BtnHavering.place(x=1210, y=320)
BtnHillingdon.place(x=285, y=360)
BtnHounslow.place(x=410, y=500)
BtnIslington.place(x=755, y=350)
BtnKensingtonAndChelsea.place(x=590, y=425)
BtnKingstonUponThames.place(x=520, y=650)
BtnLambeth.place(x=730, y=525)
BtnLewisham.place(x=870, y=550)
BtnMerton.place(x=635, y=635)
BtnNewham.place(x=940, y=390)
BtnRedbridge.place(x=990, y=270)
BtnRichmondUponThames.place(x=470, y=556)
BtnSouthwark.place(x=784, y=478)
BtnSutton.place(x=665, y=745)
BtnTowerHamlets.place(x=840, y=394)
BtnWalthamForest.place(x=873, y=235)
BtnWandsworth.place(x=625, y=548)                                                                   #All Buttons
BtnWestminster.place(x=680, y=424)

ClearButton = Button(root, text="Reset Positions", height=5, width=25, bg="#FFFFFF", fg="#000000", command=Clear)
#if Debug == True:
ClearButton.place(x=1380, y=867)

AllButtons = {
    "Barking and Dagenham": BtnBarkingAndDagenham,
    "Barnet": BtnBarnet,
    "Bexley": BtnBexley,
    "Brent": BtnBrent,
    "Bromley": BtnBromley,
    "Camden": BtnCamden,
    "City": BtnCity,
    "Croydon": BtnCroydon,
    "Ealing": BtnEaling,
    "Enfield": BtnEnfield,
    "Greenwich": BtnGreenwich,
    "Hackney": BtnHackney,
    "Hammersmith and Fulham": BtnHammersmithAndFulham,
    "Haringey": BtnHaringey,
    "Harrow": BtnHarrow,
    "Havering": BtnHavering,
    "Hillingdon": BtnHillingdon,
    "Hounslow": BtnHounslow,
    "Islington": BtnIslington,
    "Kensington and Chelsea": BtnKensingtonAndChelsea,
    "Kingston upon Thames": BtnKingstonUponThames,
    "Lambeth": BtnLambeth,
    "Lewisham": BtnLewisham,
    "Merton": BtnMerton,
    "Newham": BtnNewham,
    "Redbridge": BtnRedbridge,
    "Richmond upon Thames": BtnRichmondUponThames,
    "Southwark": BtnSouthwark,
    "Sutton": BtnSutton,
    "Tower Hamlets": BtnTowerHamlets,
    "Waltham Forest": BtnWalthamForest,
    "Wandsworth": BtnWandsworth,
    "Westminster": BtnWestminster}

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

CurrentBorough = ""
NextFile = 1                                                                                        #Main Start
Update()
    
#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.title("Rosey")                                                                             #Tkinter Windows
root.mainloop()

