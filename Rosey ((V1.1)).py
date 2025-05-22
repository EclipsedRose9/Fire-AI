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
    global FullAdress  # make sure Adress is available here if needed
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
        for entry in CurrentLine:
            if len(entry) < 2:
                continue
            UnitCode = entry[0].strip()
            UnitAddress = entry[1].strip()
            # If the entry address matches the station address, assume unit is at station.
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
                    CurrentData.append(f"{UnitNo} {TypeNeeded} from {CurrentStation['station']} (ETA {ETA} minutes).")
    else:
        pass

    # Optionally format the data into multiple Lines if desired.
    PerLine = 4  # adjust if needed
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
    global NB
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
        Transcript = open("CallTranscript"+str(NB)+".txt", "r")
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
        # In default StationInfo the station name is on line 2 and the default address is on line 3.
                StationName = Lines[1].strip()
                StationAddress = Lines[2].strip()
                DefultInfo[StationName] = StationAddress
            return DefultInfo

#################################################################################################################

        MapData = "Adjancent.txt"
        Graph = build_adjacency_Graph(MapData)
        station_file = "Stations.txt"                                                               #Update Data
        StationInfo = load_engine_data_aggregated(station_file)
        HalfPostcode = extract_half_postcode(Postcode)

#################################################################################################################

        if IncidentType == "Call Back":
            FullAdress = Adress
            StationName = CallData[1].strip()       # Station name from transcript
            UnitCode = CallData[9].strip()            # Expecting something like "P" (or "Unit: P")
            current_record = None
            for s in StationInfo:
                if s['station'] == StationName:
                    current_record = s
                    break
            if current_record is None:
                RoseyText.configure(state='normal')
                RoseyText.insert("end", f"\n...\nRosey says: No current Station found for station '{StationName}'.")
                RoseyText.configure(state='disabled')
                NB += 1
                return                                                                          

            if 'code_to_address' in current_record:
                current_unit_location = current_record['code_to_address'].get(UnitCode, current_record['StationAddress'])
            else:
                current_unit_location = current_record['StationAddress']                            #Returning Unit

            DefultInfo = GetDefult("DefultStations.txt")
            if StationName in DefultInfo:
                home_address = DefultInfo[StationName]
            else:
                home_address = current_record['StationAddress']  # fallback if not found
                
            station_half_pc = extract_half_postcode_from_address(current_unit_location)
            home_half_pc = extract_half_postcode_from_address(home_address)
            if not station_half_pc:
                station_half_pc = current_record['StationAddress'].split()[-1]
            if not home_half_pc:
                home_half_pc = home_address.split()[-1]

            Distance = compute_distance(Graph, station_half_pc, home_half_pc)
            if Distance is None:
                Distance = 2  # fallback distance
            if Distance <= 1:
                ETA_minutes = 5
            else:
                ETA_minutes = Distance * 5
            ETA_millis = ETA_minutes * 60000

            delayed_updates = [(StationName, UnitCode, None, ETA_millis)]
            schedule_station_updates(delayed_updates, home_address)

            RoseyText.configure(state='normal')
            
            dispatch_msg = (f"\n...\nRosey says: Returning unit {UnitCode} from current location {current_unit_location} to {Adress} (ETA  {ETA_minutes} minutes).\n")
            RoseyText.insert("end", dispatch_msg)
            RoseyText.configure(state='disabled')
        else:
            ImageFile = r"D:\Python Codes\Fire\Image"+str(NB)+".png"
            ImageClass = ImageAI(ImageFile)
            Logic()

#################################################################################################################
        
        NB = NB + 1
                                                       

    except Exception as e:
        print("Error in Update():", e)
        pass                                                                                          #Update Loop
    
    DisplayInfo(CurrentBorough)
    UpdateButtons()
    root.after(1000, Update)



#################################################################################################################
#################################################################################################################
#################################################################################################################

#################################################################################################################
#################################################################################################################
#################################################################################################################




###################################################
# The BFS adjacency + aggregator UnitCode (unchanged)
###################################################

def build_adjacency_Graph(adj_file_path):
    Graph = {}
    with open(adj_file_path, 'r') as File:
        for line in File:
            line = line.strip()
            if not line or ":" not in line:
                continue
            left, right = line.split(":", 1)
            node = left.strip()
            right = right.replace('.', ',')
            neighbors = [n.strip() for n in right.split(",") if n.strip()]
            if node not in Graph:
                Graph[node] = []
            Graph[node].extend(neighbors)
            for nb in neighbors:
                if nb not in Graph:
                    Graph[nb] = []
    return Graph

from collections import deque














def compute_distance(graph, start, goal):
    if start not in graph or goal not in graph:
        return None
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        current, Distance = queue.popleft()
        if current == goal:
            return Distance
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, Distance + 1))
    return None
















def extract_half_postcode(full_postcode):
    parts = full_postcode.split()
    return parts[0] if parts else full_postcode

def extract_half_postcode_from_address(address):
    tokens = address.split()
    for t in tokens:
        if 2 <= len(t) <= 4 and any(c.isdigit() for c in t):
            return t
    return None






def load_engine_data_aggregated(FileName):
    import re, ast
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
        code_array_line = Lines[3].strip()
        try:
            code_pairs = ast.literal_eval(code_array_line)
        except Exception as e:
            print(f"Error parsing Station for {StationName}: {e}")
            continue
        code_counts = {}
        engine_address_for_bfs = None
        for i, pair in enumerate(code_pairs):
            UnitCode = pair[0].strip()
            eng_addr = pair[1].strip()
            code_counts[UnitCode] = code_counts.get(UnitCode, 0) + 1
            if i == 0:
                engine_address_for_bfs = eng_addr
        StationInfo.append({
            'StationBorough': StationBorough,
            'station': StationName,
            'StationAddress': StationAddress,
            'engine_address': engine_address_for_bfs,
            'code_counts': code_counts
        })
    return StationInfo

def GetUnits(HalfPostcode, StationInfo, Graph, requirements):
    global FullAdress
    stn_info = []
    for stn in StationInfo:
        stn_half = extract_half_postcode_from_address(stn['engine_address'])
        Distance = compute_distance(Graph, stn_half, HalfPostcode)
        stn_info.append({
            'ref': stn,
            'Distance': Distance,
            'code_counts': dict(stn['code_counts'])
        })

    stn_info.sort(key=lambda x: x['Distance'] if x['Distance'] is not None else 9999)
    CurrentUnit = {}
    delayed_updates = []  # Store units with ETA for delayed station file updates

    for (TypeNeeded, qty_needed) in requirements:
        needed = qty_needed
        CurrentDispatch = []

        for info in stn_info:
            if info['Distance'] is None:
                continue
            have = info['code_counts'].get(TypeNeeded, 0)
            if have <= 0:
                continue
            take = min(have, needed)
            info['code_counts'][TypeNeeded] -= take
            needed -= take
            CurrentDispatch.append((info['ref'], take, info['Distance']))

            # Calculate ETA: if distance <= 1, ETA is 5 minutes; otherwise, ETA = (distance) * 5 minutes.
            if info['Distance'] <= 1:
                ETA = 5
            else:
                ETA = info['Distance'] * 5
            ETA_time = ETA * 60000  # convert minutes to milliseconds #######################################################################
            #ETA_time = 30000
            print(info['ref']['station'])
            print(ETA_time)

            FullAdress = str(Adress + ", " + Borough + ", " + Postcode)
            delayed_updates.append((info['ref']['station'], TypeNeeded, None, ETA_time))

            cost_here = ETA  # Use the computed ETA in minutes

            if needed <= 0:
                break

        if needed > 0:
            CurrentUnit[TypeNeeded] = None
        else:
            CurrentUnit[TypeNeeded] = CurrentDispatch

    # Schedule station updates to occur after their respective ETA
    schedule_station_updates(delayed_updates, FullAdress)

    return CurrentUnit, cost_here

def schedule_station_updates(delayed_updates, final_address):
    def scheduled_update(StationName, UnitCode, final_address, update_text):
        # Update the station file
        update_station_file("Stations.txt", [(StationName, UnitCode, final_address)])
        # Now update the text widget without deleting existing text (or clear it once at the beginning)
        RoseyText.configure(state='normal')
        RoseyText.insert("end", update_text)
        RoseyText.configure(state='disabled')

    for (StationName, UnitCode, Null, ETA_time) in delayed_updates:
        update_text = f"\n...\nRosey says: {StationName} {UnitCode} has arrived at {final_address}."
        # Immediately update to "on route"
        update_station_file("Stations.txt", [(StationName, UnitCode, f"on route to {final_address}")])
        # Schedule the final update after ETA_time milliseconds:
        root.after(ETA_time, lambda sn=StationName, uc=UnitCode, fa=final_address, ut=update_text: 
                       scheduled_update(sn, uc, fa, ut))

        
def update_station_file(FileName, dispatched_units):
    """
    Updates the station file by replacing the station address of dispatched units
    with the current incident address after their ETA has passed.

    :param FileName: The StationInfo file (e.g., "Stations.txt").
    :param dispatched_units: List of (StationName, UnitCode, new_address).
    """
    try:
        # Read the existing StationInfo file
        with open(FileName, 'r', encoding='utf-8') as File:
            FileContents = File.read().strip()

        # Split into station Stations
        Stations = re.split(r'\n\s*\n', FileContents)

        updated_records = []
        for Station in Stations:
            Lines = Station.splitlines()
            if len(Lines) < 4:
                updated_records.append(Station)
                continue

            StationBorough = Lines[0].strip()
            StationName = Lines[1].strip()
            StationAddress = Lines[2].strip()
            code_array_line = Lines[3].strip()

            # Parse the UnitCode list
            try:
                CurrentLine = ast.literal_eval(code_array_line)
            except Exception as e:
                print(f"Error parsing {StationName}: {e}")
                updated_records.append(Station)
                continue

            # Check if any dispatched unit matches this station
            for (stn_name, UnitCode, new_address) in dispatched_units:
                if StationName == stn_name:
                    # Replace the unit's address with the new incident address
                    for i, entry in enumerate(CurrentLine):
                        if entry[0] == UnitCode:
                            CurrentLine[i] = [UnitCode, new_address]  # Update address

            # Convert back to string format
            updated_code_line = str(CurrentLine)
            updated_record = f"{StationBorough}\n{StationName}\n{StationAddress}\n{updated_code_line}"
            updated_records.append(updated_record)

        # Write back to the file
        with open(FileName, 'w', encoding='utf-8') as File:
            File.write("\n\n".join(updated_records))

        print("✅ Stations file updated successfully!")

    except Exception as e:
        print(f"❌ Error updating station file: {e}")


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

#Defines the tkinter window
root = Tk()

#Text boxes
InfoBox = Text(root, state='disabled', height=65, width=45,  bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
InfoBox.place(x=1565, y=0)

RoseyText = Text(root, state='disabled', height=7, width=180, bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
RoseyText.place(x=127, y=953)

#Images
LFBMap = PhotoImage(file="LFBMap.png")
RoseyIcon = PhotoImage(file="RoseyIcon.png")

#Labels
Map = Label(root, image=LFBMap, bd=0)
Map.place(x=0, y=0)

RoseyFace = Label(root, image=RoseyIcon, bd=0)
RoseyFace.place(x=0, y=953)


#Buttons
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
BtnWandsworth.place(x=625, y=548)
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
    "Westminster": BtnWestminster,
}

CurrentBorough = ""
NB = 1
Update()
    
#Sets the tkinter window 
root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.title("Rosey")
root.mainloop()

