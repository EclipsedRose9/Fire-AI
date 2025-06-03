#################################################################################################################
#################################################################################################################
#################################################################################################################

import os
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf                                                                              #Imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import re
import ast
from collections import deque

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

Debug = False

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

# Set the data directory
TrainingData = "D:\Python Codes\Fire\FireClass"

# Define image dimensions and other parameters
ImgWidth, ImgHeight = 150, 150
batch_size = 32
NoClasses = 13

# Preprocess the images and create data generators
DataGenerator = ImageDataGenerator(rescale=1./255)

TrainGenerator = DataGenerator.flow_from_directory( TrainingData, target_size=(ImgWidth, ImgHeight), batch_size=batch_size, class_mode='categorical')

# Get the class labels from folder names
class_labels = sorted(os.listdir(TrainingData))

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(ImgWidth, ImgHeight, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),                                                          #CNN Model
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dense(NoClasses, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(TrainGenerator, epochs=10)

# Function to classify a new image
def ImageAI(ImageFile):
    CurrentImage = tf.keras.preprocessing.image.load_img(ImageFile, target_size=(ImgWidth, ImgHeight))
    ProcessingImage = tf.keras.preprocessing.image.img_to_array(CurrentImage)
    ProcessingImage = np.expand_dims(ProcessingImage, axis=0) / 255.0
    ImagePrediction = model.predict(ProcessingImage)
    Classes = np.argmax(ImagePrediction)
    ImageClass = class_labels[Classes]
    return ImageClass

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def BoroughOuput(FilePath, CurrentBorough):
    global FullAdress  # make sure Adress is available here if AmountNeeded
    global NotHome
    with open(FilePath, 'r') as File:
        FileContents = File.read().strip()

    # Split FileContents into station Stations
    Stations = re.split(r'\n\s*\n', FileContents)
    Output = []
    NotHome = 0  # Counter for units not at their station

    for Station in Stations:
        Lines = Station.splitlines()
        if len(Lines) < 4:
            continue
        
        StationBorough = Lines[0].strip()
        if StationBorough.lower() != CurrentBorough.lower():
            continue
        
        StationName = Lines[1].strip()          # Fire station name
        StationAddress = Lines[2].strip()         # Default station address                         
        
        # Parse the 2D array line containing the UnitCode/address pairs
        LineSection = Lines[3].strip()
        try:
            CurrentLine = ast.literal_eval(LineSection)                                         #Borough Display
        except:
            pass
        
        OutputContent = []
        for Entry in CurrentLine:
            if len(Entry) < 2:
                continue
            UnitCode = Entry[0].strip()
            UnitAddress = Entry[1].strip()
            # If the Entry address matches the station address, assume unit is at station.
            if UnitAddress == StationAddress:
                OutputContent.append(f"{UnitCode} at Station")
            # Otherwise, count it as not at station.
            else:
                # For entries marked "on route", we show the incident address.
                if "on route" in UnitAddress.lower():
                    OutputContent.append(f"{UnitCode} {UnitAddress}")
                else:
                    OutputContent.append(f"{UnitCode} at {UnitAddress}")
                NotHome += 1
        
        if OutputContent:
            OutputLine = StationName + "\n" + "\n".join(OutputContent)
            Output.append(OutputLine)
    
    FinalOutput = "\n\n".join(Output)
    return FinalOutput

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def Clear():
    DefultStations = open("DefultStations.txt", "r")
    ResetStations = DefultStations.read()
    DefultStations.close()
                                                                                                   #Reset Files
    CurrentStations = open("Stations.txt", "w")
    CurrentStations.write(ResetStations)
    CurrentStations.close()   

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

    
def Logic():
    Dispatch = GetPDA(IncidentType, ImageClass)
    CurrentData = []
    UnitsNeeded = []
    # Because we used 'global' in GetPDA, these variables exist now:
    if P > 0:
        UnitsNeeded.append(("P", P))
    if PL > 0:
        UnitsNeeded.append(("PL", PL))
    if FRU > 0:
        UnitsNeeded.append(("FRU", FRU))
    if Fireboat > 0:
        UnitsNeeded.append(("Fireboat", Fireboat))
    if CU > 0:
        UnitsNeeded.append(("CU", CU))
    if ALP > 0:
        UnitsNeeded.append(("ALP", ALP))
    # If you have more codes, list them here the same way.

    CurrentUnit, Null = GetUnits(HalfPostcode, StationInfo, Graph, UnitsNeeded)
    # CurrentUnit => { "P": [ (station, qty, BFSdist), ...], "PL": [...], ...}

    # Build the output string with individual ETA for each unit.
    if CurrentUnit:
        for TypeNeeded, CurrentDispatch in CurrentUnit.items():                                                #Logic Calls
            if CurrentDispatch is None:
                pass
            else:
                for (CurrentStation, UnitNo, Distance) in CurrentDispatch:
                    if Distance <= 1:
                        ETA = 5
                    else:
                        ETA = Distance * 5
                    CurrentData.append(f"{UnitNo} {TypeNeeded} from {CurrentStation['Station']} (ETA {ETA} minutes).")
    else:
        pass

    # Optionally format the data into multiple Lines if desired.
    PerLine = 4  # adjust if AmountNeeded
    Lines = []
    for i in range(0, len(CurrentData), PerLine):
        RowOutput = CurrentData[i:i+PerLine]
        Lines.append("  ".join(RowOutput))
    CurrentDataString = "\n".join(Lines)
    CurrentDataString = (f"\n...\nRosey says: There's a fire at {Adress} {Borough} {Postcode} I recommend:\n{CurrentDataString} {PDA}")
    CurrentDataString = CurrentDataString.strip()
    RoseyText.configure(state='normal')
    RoseyText.insert("end", CurrentDataString)
    RoseyText.configure(state='disable')
    # Optionally, you can call AddData() to log the data.

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################
    
def DisplayInfo(x):
    global CurrentBorough
    InfoBox.configure(state='normal')
    InfoBox.delete('1.0', END)
    InfoBox.configure(state='disabled')

    FilePath = 'Stations.txt'  # Replace with the path to your test file                         #Button Display
    CurrentBorough = x  # Change this to the StationBorough you want to find
    Output = BoroughOuput(FilePath, CurrentBorough)

    InfoBox.configure(state='normal')
    InfoBox.insert("end", "".join(Output))
    InfoBox.configure(state='disable')

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def UpdateButtons():
    
    FilePath = 'Stations.txt'
        
    for StationBorough in AllButtons:
        # Call BoroughOuput for this StationBorough.
        # This function should update the global NotHome.
        Null = BoroughOuput(FilePath, StationBorough)
        
        # Update the button's background based on the number of units not at station               #LiveDisplay
        if NotHome >= 1:
            AllButtons[StationBorough].config(bg="yellow")
        elif NotHome >= 3:
            AllButtons[StationBorough].config(bg="orange")
        elif NotHome >= 5:
            AllButtons[StationBorough].config(bg="red")
        else:
            AllButtons[StationBorough].config(bg="spring green")

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################
            
def Update():
    global NextFile
    global Adress
    global Borough
    global Postcode
    global IncidentType
    global LiveAtRisk
    global ImageClass
    global P, PL, FRU, Fireboat, CU, ALP
    global HalfPostcode
    global StationInfo
    global Graph                                                                                     #Update Try
    try:                                                                                             
        Transcript = open("CallTranscript"+str(NextFile)+".txt", "r")
        InfoBox.configure(state='normal')
        InfoBox.delete('1.0', END)
        InfoBox.configure(state='disabled')
        CallData = Transcript.readlines()
        Adress = (CallData[1].strip())
        Borough = (CallData[2].strip())
        Postcode = (CallData[3].strip())
        IncidentType = (CallData[6].strip())
        LiveAtRisk = (CallData[9].strip())

#################################################################################################################
        
        def GetDefult(FileName):
            with open(FileName, 'r', encoding='utf-8') as File:
                FileContents = File.read().strip()
            Stations = re.split(r'\n\s*\n', FileContents)
            DefultInfo = {}
            for Row in Stations:
                Lines = Row.splitlines()
                if len(Lines) < 3:                                                              #Defult Stations
                    continue
        # In default StationInfo the station nam
                StationName = Lines[1].strip()
                StationAddress = Lines[2].strip()
                DefultInfo[StationName] = StationAddress
            return DefultInfo

#################################################################################################################

        MapData = "Adjancent.txt"
        Graph = GraphAdjacency(MapData)
        StationFile = "Stations.txt"                                                               #Update Data
        StationInfo = SplitStations(StationFile)
        HalfPostcode = GetHalf(Postcode)

#################################################################################################################

        if IncidentType == "Call Back":
            FullAdress = Adress
            StationName = CallData[1].strip()       # Station name from transcript
            UnitCode = CallData[9].strip()            # Expecting something like "P" (or "Unit: P")
            CurrentRecord = None
            for NextStation in StationInfo:
                if NextStation['Station'] == StationName:
                    CurrentRecord = NextStation
                    break
            if CurrentRecord is None:
                RoseyText.configure(state='normal')
                RoseyText.insert("end", f"\n...\nRosey says: No current Station found for Station '{StationName}'.")
                RoseyText.configure(state='disabled')
                NextFile += 1
                return                                                                          

            if 'CodeAddress' in CurrentRecord:
                CurrentLocation = CurrentRecord['CodeAddress'].get(UnitCode, CurrentRecord['StationAddress'])
            else:
                CurrentLocation = CurrentRecord['StationAddress']                            #Returning Unit

            DefultInfo = GetDefult("DefultStations.txt")
            if StationName in DefultInfo:
                HomeAddress = DefultInfo[StationName]
            else:
                HomeAddress = CurrentRecord['StationAddress']  # fallback if not found
                
            StationHalf = ReturnHalf(CurrentLocation)
            HomeHalf = ReturnHalf(HomeAddress)
            if not StationHalf:
                StationHalf = CurrentRecord['StationAddress'].split()[-1]
            if not HomeHalf:
                HomeHalf = HomeAddress.split()[-1]

            Distance = GetDistance(Graph, StationHalf, HomeHalf)
            if Distance is None:
                Distance = 2  # fallback distance
            if Distance <= 1:
                ETAMins = 5
            else:
                ETAMins = Distance * 5
            ETAMili = ETAMins * 60000

            DelayedUpdate = [(StationName, UnitCode, None, ETAMili)]
            ScheduledUpdates(DelayedUpdate, HomeAddress)

            RoseyText.configure(state='normal')
            
            DispatchOutput = (f"\n...\nRosey says: Returning unit {UnitCode} from current location {CurrentLocation} to {Adress} (ETA  {ETAMins} minutes).\n")
            RoseyText.insert("end", DispatchOutput)
            RoseyText.configure(state='disabled')
        else:
            ImageFile = r"D:\Python Codes\Fire\Image"+str(NextFile)+".png"
            ImageClass = ImageAI(ImageFile)
            Logic()

#################################################################################################################
        
        NextFile = NextFile + 1
                                                       

    except:
        pass                                                                                        #Update Loop                                                                                         
                                                                                                    
    DisplayInfo(CurrentBorough)                                                                     
    UpdateButtons()
    root.after(1000, Update)

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def GraphAdjacency(AdjancencyPath):
    Graph = {}
    with open(AdjancencyPath, 'r') as File:
        for line in File:
            line = line.strip()
            if not line or ":" not in line:
                continue
            Left, Right = line.split(":", 1)
            Node = Left.strip()
            Right = Right.replace('.', ',')                                                         #Build Graph
            Adjancent = [n.strip() for n in Right.split(",") if n.strip()]
            if Node not in Graph:
                Graph[Node] = []
            Graph[Node].extend(Adjancent)
            for Neighbor in Adjancent:
                if Neighbor not in Graph:
                    Graph[Neighbor] = []
    return Graph

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def GetDistance(Graph, Start, End):
    if Start not in Graph or End not in Graph:
        return None
    Visited = set()
    Queue = deque([(Start, 0)])
    Visited.add(Start)
    while Queue:
        Current, Distance = Queue.popleft()                                                         #Get Distance
        if Current == End:
            return Distance
        for Neighbor in Graph[Current]:
            if Neighbor not in Visited:
                Visited.add(Neighbor)
                Queue.append((Neighbor, Distance + 1))
    return None

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def GetHalf(FullPostcode):
    Parts = FullPostcode.split()
    return Parts[0] if Parts else FullPostcode

def ReturnHalf(FullPostcode):                                                                    #Split Postcodes
    Tokens = FullPostcode.split()
    for t in Tokens:
        if 2 <= len(t) <= 4 and any(c.isdigit() for c in t):
            return t
    return None

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def SplitStations(FileName):
    with open(FileName, 'r', encoding='utf-8') as File:
        FileContents = File.read().strip()
    Stations = re.split(r'\n\s*\n', FileContents)
    StationInfo = []
    for Row in Stations:
        Lines = Row.splitlines()
        if len(Lines) < 4:
            continue
        StationBorough = Lines[0].strip()
        StationName = Lines[1].strip()
        StationAddress = Lines[2].strip()
        CodeLine = Lines[3].strip()
        try:
            Pairs = ast.literal_eval(CodeLine)
        except:                                                                      #Station Info
            pass
        UnitNo = {}
        UnitLocation = None
        for i, Pair in enumerate(Pairs):
            UnitCode = Pair[0].strip()
            UnitAddress = Pair[1].strip()
            UnitNo[UnitCode] = UnitNo.get(UnitCode, 0) + 1
            if i == 0:
                UnitLocation = UnitAddress
        StationInfo.append({
            'StationBorough': StationBorough,
            'Station': StationName,
            'StationAddress': StationAddress,
            'EngineAdress': UnitLocation,
            'UnitNo': UnitNo})
        
    return StationInfo

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def GetUnits(HalfPostcode, StationInfo, Graph, Requirements):
    global FullAdress
    StationData = []
    for Station in StationInfo:
        StationHalf = ReturnHalf(Station['EngineAdress'])
        Distance = GetDistance(Graph, StationHalf, HalfPostcode)
        StationData.append({'Station': Station, 'Distance': Distance, 'UnitNo': dict(Station['UnitNo'])})

    StationData.sort(key=lambda x: x['Distance'] if x['Distance'] is not None else 9999)
    CurrentUnit = {}
    DelayedUpdate = []  # Store units with ETA for delayed station file updates

    for (TypeNeeded, QuantityNeeded) in Requirements:
        AmountNeeded = QuantityNeeded
        CurrentDispatch = []

        for Info in StationData:
            if Info['Distance'] is None:
                continue
            Got = Info['UnitNo'].get(TypeNeeded, 0)
            if Got <= 0:
                continue
            Take = min(Got, AmountNeeded)
            Info['UnitNo'][TypeNeeded] -= Take
            AmountNeeded -= Take
            CurrentDispatch.append((Info['Station'], Take, Info['Distance']))                           
                                                                                                    #Get Dispatch
            if Info['Distance'] <= 1:
                ETA = 5
            else:
                ETA = Info['Distance'] * 5
            ETA_time = ETA * 60000  # convert minutes to milliseconds ######################
            if Debug == True:
                ETA_time = 10000
            print(Info['Station']['Station'])
            print(ETA_time)

            FullAdress = str(Adress + ", " + Borough + ", " + Postcode)
            DelayedUpdate.append((Info['Station']['Station'], TypeNeeded, None, ETA_time))

            cost_here = ETA  # Use the computed ETA in minutes

            if AmountNeeded <= 0:
                break

        if AmountNeeded > 0:
            CurrentUnit[TypeNeeded] = None
        else:
            CurrentUnit[TypeNeeded] = CurrentDispatch

    ScheduledUpdates(DelayedUpdate, FullAdress)
    return CurrentUnit, cost_here

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################

def ScheduledUpdates(DelayedUpdate, FinalAddress):
    def ScheduledUpdate(StationName, UnitCode, FinalAddress, UpdatedText):
        # Update the station file
        UpdateFile("Stations.txt", [(StationName, UnitCode, FinalAddress)])
        # Now update the text widget without deleting existing text (or clear it once at the beginning)
        RoseyText.configure(state='normal')
        RoseyText.insert("end", UpdatedText)
        RoseyText.configure(state='disabled')                                                    #Planned Updates
                                                                                                
    for (StationName, UnitCode, Null, ETA_time) in DelayedUpdate:
        UpdatedText = f"\n...\nRosey says: {StationName} {UnitCode} has arrived at {FinalAddress}."
        # Immediately update to "on route"
        UpdateFile("Stations.txt", [(StationName, UnitCode, f"on route to {FinalAddress}")])
        # Schedule the final update after ETA_time milliseconds:
        root.after(ETA_time, lambda SN=StationName, UC=UnitCode, FA=FinalAddress, UT=UpdatedText: ScheduledUpdate(SN, UC, FA, UT))

#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################
        
def UpdateFile(FileName, DispatchedUnits):

    try:
        with open(FileName, 'r', encoding='utf-8') as File:
            FileContents = File.read().strip()

        Stations = re.split(r'\n\s*\n', FileContents)

        UpdatedInfo = []
        for Station in Stations:
            Lines = Station.splitlines()
            if len(Lines) < 4:
                UpdatedInfo.append(Station)
                continue

            StationBorough = Lines[0].strip()
            StationName = Lines[1].strip()
            StationAddress = Lines[2].strip()
            CodeLine = Lines[3].strip()

            try:
                CurrentLine = ast.literal_eval(CodeLine)
            except:
                UpdatedInfo.append(Station)
                continue

            for (Station, UnitCode, NewAddress) in DispatchedUnits:
                if StationName == Station:
                    
                    for i, Entry in enumerate(CurrentLine):
                        if Entry[0] == UnitCode:
                            CurrentLine[i] = [UnitCode, NewAddress]  # Update address

            UpdatedLine = str(CurrentLine)
            UpdatedRecord = f"{StationBorough}\n{StationName}\n{StationAddress}\n{UpdatedLine}"
            UpdatedInfo.append(UpdatedRecord)

        with open(FileName, 'w', encoding='utf-8') as File:
            File.write("\n\n".join(UpdatedInfo))
    except:
        pass

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
    print(P)
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

