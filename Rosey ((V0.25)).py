import os
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import re
import ast

# Set the data directory
data_dir = "D:\Python Codes\Fire\FireClass"

# Define image dimensions and other parameters
img_width, img_height = 150, 150
batch_size = 32
num_classes = 13

# Preprocess the images and create data generators
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Get the class labels from folder names
class_labels = sorted(os.listdir(data_dir))

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Function to classify a new image
def classify_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    return predicted_class


def extract_and_process_stations(file_path, target_borough):
    global FullAdress  # make sure Adress is available here if needed
    global total_units_not_at_station
    with open(file_path, 'r') as f:
        content = f.read().strip()

    # Split content into station records
    records = re.split(r'\n\s*\n', content)
    output_sections = []
    total_units_not_at_station = 0  # Counter for units not at their station

    for record in records:
        lines = record.splitlines()
        if len(lines) < 4:
            continue
        
        borough = lines[0].strip()
        if borough.lower() != target_borough.lower():
            continue
        
        station_name = lines[1].strip()          # Fire station name
        station_address = lines[2].strip()         # Default station address
        
        # Parse the 2D array line containing the code/address pairs
        array_line = lines[3].strip()
        try:
            code_entries = ast.literal_eval(array_line)
        except Exception as e:
            print(f"Error parsing record starting with '{station_name}': {e}")
            continue
        
        valid_codes = []
        for entry in code_entries:
            if len(entry) < 2:
                continue
            code = entry[0].strip()
            entry_address = entry[1].strip()
            # If the entry address matches the station address, assume unit is at station.
            if entry_address == station_address:
                valid_codes.append(f"{code} at Station")
            # Otherwise, count it as not at station.
            else:
                # For entries marked "on route", we show the incident address.
                if entry_address.lower() == "on route":
                    valid_codes.append(f"{code} on route to {FullAdress}")
                else:
                    valid_codes.append(f"{code} at {entry_address}")
                total_units_not_at_station += 1
        
        if valid_codes:
            section_output = station_name + "\n" + "\n".join(valid_codes)
            output_sections.append(section_output)
    
    final_output = "\n\n".join(output_sections)
    return final_output

def Clear():
    DefultStations = open("DefultStations.txt", "r")
    ResetStations = DefultStations.read()
    DefultStations.close()

    CurrentStations = open("Stations.txt", "w")
    CurrentStations.write(ResetStations)
    CurrentStations.close()

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

    
    
def Logic():
    Dispatch = GetPDA(IncidentType, predicted_class)
    print(PDA)
    CurrentData = []
    units_required = []
    # Because we used 'global' in GetPDA, these variables exist now:
    if P > 0:
        units_required.append(("P", P))
    if PL > 0:
        units_required.append(("PL", PL))
    if FRU > 0:
        units_required.append(("FRU", FRU))
    if Fireboat > 0:
        units_required.append(("Fireboat", Fireboat))
    if CU > 0:
        units_required.append(("CU", CU))
    if ALP > 0:
        units_required.append(("ALP", ALP))
    # If you have more codes, list them here the same way.

    usage, _ = find_all_units(user_half_postcode, stations, graph, units_required)
    # usage => { "P": [ (station, qty, BFSdist), ...], "PL": [...], ...}

    # Build the output string with individual ETA for each unit.
    if usage:
        for code_needed, used_list in usage.items():
            if used_list is None:
                print(f"Not enough stations to fulfill {code_needed}")
            else:
                for (stn_obj, qty_taken, dist) in used_list:
                    if dist <= 1:
                        eta = 5
                    else:
                        eta = dist * 5
                    CurrentData.append(f"{qty_taken} {code_needed} from {stn_obj['station']} (ETA {eta} minutes).")
    else:
        print("No usage results (maybe no codes were needed).")

    # Optionally format the data into multiple lines if desired.
    items_per_line = 4  # adjust if needed
    lines = []
    for i in range(0, len(CurrentData), items_per_line):
        row_items = CurrentData[i:i+items_per_line]
        lines.append("  ".join(row_items))
    CurrentDataString = "\n".join(lines)

    CurrentDataString = ("\nRosey says: There's a fire at " + Adress + " " + Borough + " " + Postcode +
                         " I recommend:\n" + CurrentDataString +" "+ PDA +"\n...")
    CurrentDataString = CurrentDataString.strip()
    RoseyText.configure(state='normal')
    RoseyText.insert("end", CurrentDataString)
    RoseyText.configure(state='disable')
    # Optionally, you can call AddData() to log the data.

def AddData(x, y):
    CurrentFile = open(str(x)+".txt", "a")
    CurrentFile.write(y+"\n\n")
    CurrentFile.close()

    AllFile = open("All.txt", "a")
    AllFile.write(y+"\n\n")
    AllFile.close()
    
def DisplayInfo(x):
    global target_borough
    InfoBox.configure(state='normal')
    InfoBox.delete('1.0', END)
    InfoBox.configure(state='disabled')

    if __name__ == '__main__':
        file_path = 'Stations.txt'  # Replace with the path to your test file
        target_borough = x  # Change this to the borough you want to find
        result = extract_and_process_stations(file_path, target_borough)

    InfoBox.configure(state='normal')
    InfoBox.insert("end", "".join(result))
    InfoBox.configure(state='disable')

def UpdateButtons():
    
    file_path = 'Stations.txt'
        
    for borough in AllButtons:
        # Call extract_and_process_stations for this borough.
        # This function should update the global total_units_not_at_station.
        _ = extract_and_process_stations(file_path, borough)
        
        # Update the button's background based on the number of units not at station
        if total_units_not_at_station >= 1:
            AllButtons[borough].config(bg="yellow")
        elif total_units_not_at_station >= 3:
            AllButtons[borough].config(bg="orange")
        elif total_units_not_at_station >= 5:
            AllButtons[borough].config(bg="red")
        else:
            AllButtons[borough].config(bg="spring green")
            
def Update():
    global NB
    global Adress
    global Borough
    global Postcode
    global IncidentType
    global LiveAtRisk
    global predicted_class
    global P, PL, FRU, Fireboat, CU, ALP
    global user_half_postcode
    global stations
    global graph
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
        def extract_half_postcode(full_postcode):
            parts = full_postcode.split()
            return parts[0] if parts else full_postcode

        adjacency_file = "Adjancent.txt"
        graph = build_adjacency_graph(adjacency_file)
        
        station_file = "Stations.txt"
        stations = load_engine_data_aggregated(station_file)

        # ------------------------------------------------------------------
        # 5) BFS aggregator to find the needed codes from nearest stations
        # ------------------------------------------------------------------
        user_half_postcode = extract_half_postcode(Postcode)








        def load_default_stations(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            records = re.split(r'\n\s*\n', content)
            default_stations = {}
            for rec in records:
                lines = rec.splitlines()
                if len(lines) < 3:
                    continue
        # In default stations the station name is on line 2 and the default address is on line 3.
                station_name = lines[1].strip()
                station_address = lines[2].strip()
                default_stations[station_name] = station_address
            return default_stations

# ----------------- Revised "Call Back" branch -----------------
        if IncidentType == "Call Back":
    # Assume the new transcript format:
    #   Line 0: "Address:" 
    #   Line 1: The station name (i.e. the home station)
    #   ...
    #   A later line (e.g. line 7 or 8) contains "Unit:" followed by the unit code.
    # Adjust the indexes below as needed based on your transcript format.
            station_name = CallData[1].strip()       # Station name from transcript
            unit_line = CallData[9].strip()            # Expecting something like "P" (or "Unit: P")
            if "Unit:" in unit_line:
                unit_code = unit_line.split("Unit:")[-1].strip()
            else:
                unit_code = unit_line

    # Using the station name, look up the current station record from the "stations" data.
    # (stations is assumed to be loaded already from "Stations.txt")
            current_record = None
            for s in stations:
                if s['station'] == station_name:
                    current_record = s
                    break
            if current_record is None:
                RoseyText.configure(state='normal')
                RoseyText.insert("end", f"\nRosey says: No current record found for station '{station_name}'.\n")
                RoseyText.configure(state='disabled')
                NB += 1
                return

    # Determine the current location of the unit.
    # Here we assume you have enhanced your station loading so that each record includes a
    # dictionary 'code_to_address' mapping each unit code to its current location.
    # If not, you can fall back to the station's default address.
            if 'code_to_address' in current_record:
                current_unit_location = current_record['code_to_address'].get(unit_code, current_record['station_address'])
            else:
                current_unit_location = current_record['station_address']

    # Now load the default stations data to get the home address for this station.
            default_stations = load_default_stations("DefultStations.txt")
            if station_name in default_stations:
                home_address = default_stations[station_name]
            else:
                home_address = current_record['station_address']  # fallback if not found

    # For route calculation, we need to extract half-postcodes.
    # Use the current unit location (from stations) as the starting point
    # and the home_address (from default stations) as the destination.
            station_half_pc = extract_half_postcode_from_address(current_unit_location)
            home_half_pc = extract_half_postcode_from_address(home_address)
    # If extraction fails, you may choose to default to a region code.
            if not station_half_pc:
                station_half_pc = current_record['station_address'].split()[-1]
            if not home_half_pc:
                home_half_pc = home_address.split()[-1]

    # Compute BFS distance between the current location and the home station.
            dist = compute_distance(graph, station_half_pc, home_half_pc)
            if dist is None:
                dist = 2  # fallback distance
            if dist <= 1:
                eta_minutes = 5
            else:
                eta_minutes = dist * 5
            eta_millis = eta_minutes * 60000

    # Schedule the station update: after ETA the unit will be marked as returned (i.e. at its home station address).
            delayed_updates = [(station_name, unit_code, None, eta_millis)]
            schedule_station_updates(delayed_updates, home_address)

    # Log the dispatch in the GUI.
            RoseyText.configure(state='normal')

            
            dispatch_msg = (f"\nRosey says: Returning unit {unit_code} from current location {current_unit_location} to {Adress} (ETA  {eta_minutes} minutes).\n...")
            RoseyText.insert("end", dispatch_msg)
            RoseyText.configure(state='disabled')

        else:
            image_path = r"D:\Python Codes\Fire\Image"+str(NB)+".png"
            predicted_class = classify_image(image_path)
    

























        
        NB = NB + 1
        Logic()  # <= your function that calls GetPDA internally

        # ------------------------------------------------------------------
        # 3) Build a BFS-based adjacency graph & aggregated station data
        # ------------------------------------------------------------------
        


        # ------------------------------------------------------------------
        # 7) Show the existing logs, close out
        # ------------------------------------------------------------------
        AllFile = open("All.txt", "r")
        AllData = AllFile.readlines()
        AllFile.close()
        InfoBox.configure(state='normal')
        InfoBox.insert("end", "".join(AllData))
        InfoBox.configure(state='disable')

    except Exception as e:
        print("Error in Update():", e)
        pass
    
    DisplayInfo(target_borough)
    UpdateButtons()
    root.after(1000, Update)


###################################################
# The BFS adjacency + aggregator code (unchanged)
###################################################

def build_adjacency_graph(adj_file_path):
    graph = {}
    with open(adj_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            left, right = line.split(":", 1)
            node = left.strip()
            right = right.replace('.', ',')
            neighbors = [n.strip() for n in right.split(",") if n.strip()]
            if node not in graph:
                graph[node] = []
            graph[node].extend(neighbors)
            for nb in neighbors:
                if nb not in graph:
                    graph[nb] = []
    return graph

from collections import deque

def compute_distance(graph, start, goal):
    if start not in graph or goal not in graph:
        return None
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        current, dist = queue.popleft()
        if current == goal:
            return dist
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
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

def load_engine_data_aggregated(filename):
    import re, ast
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    records = re.split(r'\n\s*\n', content)
    stations = []
    for rec in records:
        lines = rec.splitlines()
        if len(lines) < 4:
            continue
        borough = lines[0].strip()
        station_name = lines[1].strip()
        station_address = lines[2].strip()
        code_array_line = lines[3].strip()
        try:
            code_pairs = ast.literal_eval(code_array_line)
        except Exception as e:
            print(f"Error parsing record for {station_name}: {e}")
            continue
        code_counts = {}
        engine_address_for_bfs = None
        for i, pair in enumerate(code_pairs):
            code = pair[0].strip()
            eng_addr = pair[1].strip()
            code_counts[code] = code_counts.get(code, 0) + 1
            if i == 0:
                engine_address_for_bfs = eng_addr
        stations.append({
            'borough': borough,
            'station': station_name,
            'station_address': station_address,
            'engine_address': engine_address_for_bfs,
            'code_counts': code_counts
        })
    return stations

def find_all_units(user_half_postcode, stations, graph, requirements):
    global FullAdress
    stn_info = []
    for stn in stations:
        stn_half = extract_half_postcode_from_address(stn['engine_address'])
        dist = compute_distance(graph, stn_half, user_half_postcode)
        stn_info.append({
            'ref': stn,
            'dist': dist,
            'code_counts': dict(stn['code_counts'])
        })

    stn_info.sort(key=lambda x: x['dist'] if x['dist'] is not None else 9999)
    usage = {}
    delayed_updates = []  # Store units with ETA for delayed station file updates

    for (code_needed, qty_needed) in requirements:
        needed = qty_needed
        used_list = []

        for info in stn_info:
            if info['dist'] is None:
                continue
            have = info['code_counts'].get(code_needed, 0)
            if have <= 0:
                continue
            take = min(have, needed)
            info['code_counts'][code_needed] -= take
            needed -= take
            used_list.append((info['ref'], take, info['dist']))

            # Calculate ETA: if distance <= 1, ETA is 5 minutes; otherwise, ETA = (distance) * 5 minutes.
            if info['dist'] <= 1:
                eta = 5
            else:
                eta = info['dist'] * 5
            eta_time = eta * 60000  # convert minutes to milliseconds #######################################################################
            #eta_time = 30000
            print(info['ref']['station'])
            print(eta_time)

            FullAdress = str(Adress + ", " + Borough + ", " + Postcode)
            delayed_updates.append((info['ref']['station'], code_needed, None, eta_time))

            cost_here = eta  # Use the computed eta in minutes

            if needed <= 0:
                break

        if needed > 0:
            usage[code_needed] = None
        else:
            usage[code_needed] = used_list

    # Schedule station updates to occur after their respective ETA
    schedule_station_updates(delayed_updates, FullAdress)

    return usage, cost_here

def schedule_station_updates(delayed_updates, final_address):
    def scheduled_update(station_name, unit_code, final_address, update_text):
        # Update the station file
        update_station_file("Stations.txt", [(station_name, unit_code, final_address)])
        # Now update the text widget without deleting existing text (or clear it once at the beginning)
        RoseyText.configure(state='normal')
        RoseyText.insert("end", update_text)
        RoseyText.configure(state='disabled')

    for (station_name, unit_code, _, eta_time) in delayed_updates:
        update_text = f"\nRosey says: {station_name} {unit_code} has arrived at {final_address}\n..."
        # Immediately update to "on route"
        update_station_file("Stations.txt", [(station_name, unit_code, "on route")])
        # Schedule the final update after eta_time milliseconds:
        root.after(eta_time, lambda sn=station_name, uc=unit_code, fa=final_address, ut=update_text: 
                       scheduled_update(sn, uc, fa, ut))

        
def update_station_file(filename, dispatched_units):
    """
    Updates the station file by replacing the station address of dispatched units
    with the current incident address after their ETA has passed.

    :param filename: The stations file (e.g., "Stations.txt").
    :param dispatched_units: List of (station_name, unit_code, new_address).
    """
    try:
        # Read the existing stations file
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Split into station records
        records = re.split(r'\n\s*\n', content)

        updated_records = []
        for record in records:
            lines = record.splitlines()
            if len(lines) < 4:
                updated_records.append(record)
                continue

            borough = lines[0].strip()
            station_name = lines[1].strip()
            station_address = lines[2].strip()
            code_array_line = lines[3].strip()

            # Parse the code list
            try:
                code_entries = ast.literal_eval(code_array_line)
            except Exception as e:
                print(f"Error parsing {station_name}: {e}")
                updated_records.append(record)
                continue

            # Check if any dispatched unit matches this station
            for (stn_name, unit_code, new_address) in dispatched_units:
                if station_name == stn_name:
                    # Replace the unit's address with the new incident address
                    for i, entry in enumerate(code_entries):
                        if entry[0] == unit_code:
                            code_entries[i] = [unit_code, new_address]  # Update address

            # Convert back to string format
            updated_code_line = str(code_entries)
            updated_record = f"{borough}\n{station_name}\n{station_address}\n{updated_code_line}"
            updated_records.append(updated_record)

        # Write back to the file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(updated_records))

        print("✅ Stations file updated successfully!")

    except Exception as e:
        print(f"❌ Error updating station file: {e}")

#Defines the tkinter window
root = Tk()

#Text boxes
InfoBox = Text(root, state='disabled', height=65, width=45,  bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
InfoBox.place(x=1565, y=0)

RoseyText = Text(root, state='disabled', height=7.5, width=180, bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
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

target_borough = ""
NB = 1
Update()
    
#Sets the tkinter window 
root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.title("Rosey")
root.mainloop()

