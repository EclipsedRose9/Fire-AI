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
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    # Split content into station records
    records = re.split(r'\n\s*\n', content)
    output_sections = []
    
    for record in records:
        lines = record.splitlines()
        if len(lines) < 4:
            continue
        
        borough = lines[0].strip()
        if borough.lower() != target_borough.lower():
            continue
        
        station_name = lines[1].strip()  # Fire station name
        station_address = lines[2].strip()  # Original station address
        
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
            # If the entry address matches the original station address, show "at Station"
            if entry_address == station_address:
                valid_codes.append(f"{code} at Station")
            # If it's marked as "on route", include the incident address in the output.
            elif entry_address.lower() == "on route":
                # You can use just Adress or a full incident address like f"{Adress}, {Borough}, {Postcode}"
                valid_codes.append(f"{code} on route to {FullAdress}")
            else:
                valid_codes.append(f"{code} ({entry_address})")
        
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
    global P, PL, FRU, Fireboat, CU, ALP
    if str(x) == "House Fire" and str(LiveAtRisk) == "No":
        PDA = ""+y+" P, 1 PL and 1 officer in charge"
        P = y
        PL = 1
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
        
    elif str(x) == "House Fire" and str(LiveAtRisk) == "Yes":
        PDA = ""+y+" P, 1 PL and 1 officer in charge, ambulance notified"
        P = y
        PL = 1
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "High-Rise Fire":
        PDA = ""+y+" P, 1 ALP, 1 CU and multiple officers"
        P = y
        PL = 1
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Large Vehicle Fire":
        PDA = "2 P, 1 PL and 1 officer in charge"
        P = 2
        PL = 1
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Bin/Small Rubbish Fire":
        PDA = "1 P"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Cladding Fire (High-Rise)":
        PDA = ""+y+" P, 3 PL, multiple officers and CU"
        P = y
        PL = 3
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    elif str(x) == "Fire with Fatality":
        PDA = ""+y+" P, 2 FIU and press officer notified"
        P = y
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Fire Survival Guidance (FSG) Calls":
        PDA = "Additional CU and personnel dispatched for complex life-risk situations"
        P = 0
        PL = 0
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    
    # Special Services (Rescues & Non-Fire Emergencies)
    elif str(x) == "Person Trapped (Non-Road Traffic Collision)":
        PDA = "2 P, 1 FRU and 1 officer in charge"
        P = 2
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 1
        ALP = 0
    elif str(x) == "Road Traffic Collision (Person Trapped)":
        PDA = "2 P, 1 FRU, 1 officer in charge, ambulance notified"
        P = 2
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 1
        ALP = 0
    elif str(x) == "Person Shut in Lift (Non-Emergency)":
        PDA = "1 P (only if no other help available)"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Animal Rescue (Large Animal)":
        PDA = "1 P, 1 FRU and specialist officers"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 1
        ALP = 0
    elif str(x) == "Flooding (Non-Commercial)":
        PDA = "1 P"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    
    # Hazardous Materials (HazMat) Incidents
    elif str(x) == "Gas Leak (Domestic/Commercial)":
        PDA = "1 P and gas authority notified"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Minor Hazardous Spill (under 100L)":
        PDA = "1 P"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Major Hazardous Spill (over 100L)":
        PDA = "2 P, HazMat unit and CU"
        P = 2
        PL = 0
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    elif str(x) == "Suspicious Substance/White Powder":
        PDA = "1 HazMat officer and police notified"
        P = 0
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Deliberate HazMat Release":
        PDA = "6 P, multiple CU, HazMat and specialist units"
        P = 6
        PL = 0
        Fireboat = 0
        CU = 2
        FRU = 0
        ALP = 0
    
    # Other Emergencies
    elif str(x) == "Person Threatening to Jump/Rescue from Height":
        PDA = "1 P, 1 PL, ambulance and police notified"
        P = 1
        PL = 1
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Person Collapsed Behind Locked Door":
        PDA = "1 P and ambulance notified"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Vehicle into Building":
        PDA = "2 P, 1 FRU, 1 structural officer and ambulance"
        P = 2
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 1
        ALP = 0
    elif str(x) == "Train or Tram Crash":
        PDA = "4 P, 1 CU, 1 FRU, 1 USAR team"
        P = 4
        PL = 0
        Fireboat = 0
        CU = 1
        FRU = 1
        ALP = 0
    elif str(x) == "Aircraft Accident":
        PDA = ""+y+" P, CU, specialist officers"
        P = y
        PL = 0
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    elif str(x) == "Fire on a Vessel":
        PDA = "3 P, a Fireboat, CU support"
        P = 3
        PL = 0
        Fireboat = 1
        CU = 1
        FRU = 0
        ALP = 0
        
    
    # Additional Response Triggers
    elif str(x) == "Multiple Calls (4 or more)":
        PDA = "Additional CU and station commander dispatched"
        P = 0
        PL = 0
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    elif str(x) == "High-Rise Cladding Fire":
        PDA = "8 P, multiple PL, CU and senior officers"
        P = 8
        PL = 2
        Fireboat = 0
        CU = 1
        FRU = 0
        ALP = 0
    elif str(x) == "Confirmed Life Risk":
        PDA = "Additional P and ambulance dispatched immediately"
        P = 1
        PL = 0
        Fireboat = 0
        CU = 0
        FRU = 0
        ALP = 0
    elif str(x) == "Large Fires (6+ P)":
        PDA = "Extra CU, fire safety officers and senior officers dispatched"
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
    PDA = GetPDA(IncidentType, predicted_class)
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

    usage, cost_here = find_all_units(user_half_postcode, stations, graph, units_required)
        # usage => { "P": [ (station, qty, BFSdist), ...], "PL": [...], ...}
        # total_time => naive sum of BFS-based times

        # ------------------------------------------------------------------
        # 6) Print or store the results (if usage is not empty)
        # ------------------------------------------------------------------
    if usage:
            # If we found stations for those codes, show them
            # e.g. a quick print out:
        for code_needed, used_list in usage.items():
            if used_list is None:
                print(f"Not enough stations to fulfill {code_needed}")
            else:
                for (stn_obj, qty_taken, dist) in used_list:
                    CurrentData.append("" + str(qty_taken) +" "+ str(code_needed) + " from " + str(stn_obj['station']) + " (ETA " + str(cost_here) + " minuets). ")
    else:
        print("No usage results (maybe no codes were needed).")

    items_per_line = 4  # Change to 4 if desired

    lines = []
    for i in range(0, len(CurrentData), items_per_line):
        # Slice a chunk of items_per_line items
        row_items = CurrentData[i:i+items_per_line]
        # Join them with a space (or any separator you prefer)
        lines.append("  ".join(row_items))

    CurrentDataString = "\n".join(lines)

    CurrentDataString = ("Rosey says: Theres a fire at "+Adress+" "+Borough+" "+Postcode+" I recommend:\n "+str(CurrentDataString))
    CurrentDataString = CurrentDataString.strip()
    RoseyText.configure(state='normal')
    RoseyText.insert("end", "".join(CurrentDataString))
    RoseyText.configure(state='disable')
    #AddData(Borough, CurrentData)

def AddData(x, y):
    CurrentFile = open(str(x)+".txt", "a")
    CurrentFile.write(y+"\n\n")
    CurrentFile.close()

    AllFile = open("All.txt", "a")
    AllFile.write(y+"\n\n")
    AllFile.close()
    
def DisplayInfo(x):
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
        RoseyText.configure(state='normal')
        RoseyText.delete('1.0', END)
        RoseyText.configure(state='disabled')
        image_path = r"D:\Python Codes\Fire\Image"+str(NB)+".png"
        NB = NB + 1
        predicted_class = classify_image(image_path)
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
    """
    Finds the closest stations that can fulfill the requested unit types
    and delays updating `Stations.txt` until the ETA time has passed.
    """
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

            # ✅ Calculate ETA (hops * 5 minutes, converted to milliseconds for Tkinter)
            eta_time = (info['dist'] + 1) * 5 * 60000  # Convert to milliseconds
            
            # ✅ Store the station update info with delay
            FullAdress = str(Adress +", "+ Borough +", "+ Postcode)
            delayed_updates.append((info['ref']['station'], code_needed, None, eta_time))

            cost_here = (info['dist'] + 1) * 5  # Convert hops to minutes

            if needed <= 0:
                break

        if needed > 0:
            usage[code_needed] = None
        else:
            usage[code_needed] = used_list

    # ✅ Schedule station updates to occur after their respective ETA
    schedule_station_updates(delayed_updates, FullAdress)

    return usage, cost_here

def schedule_station_updates(delayed_updates, final_address):
    """
    Delays updating the stations file until each unit's ETA has passed.
    """
    for (station_name, unit_code, _, eta_time) in delayed_updates:
        # Immediately update to "on route"
        update_station_file("Stations.txt", [(station_name, unit_code, "on route")])
        # Schedule the final update after eta_time milliseconds:
        root.after(eta_time, lambda sn=station_name, uc=unit_code, na=final_address: 
                   update_station_file("Stations.txt", [(sn, uc, na)]))
        
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
Btn1 = Button(root, text="Barking and\nDagenham", command=lambda: DisplayInfo("Barking and Dagenham"))
Btn2 = Button(root, text="Barnet", command=lambda: DisplayInfo("Barnet"))
Btn3 = Button(root, text="Bexley", command=lambda: DisplayInfo("Bexley"))
Btn4 = Button(root, text="Brent", command=lambda: DisplayInfo("Brent"))
Btn5 = Button(root, text="Bromley", command=lambda: DisplayInfo("Bromley"))
Btn6 = Button(root, text="Camden", command=lambda: DisplayInfo("Camden"))
Btn7 = Button(root, text="City", command=lambda: DisplayInfo("City"))
Btn8 = Button(root, text="Croydon", command=lambda: DisplayInfo("Croydon"))
Btn9 = Button(root, text="Ealing", command=lambda: DisplayInfo("Ealing"))
Btn10 = Button(root, text="Enfield", command=lambda: DisplayInfo("Enfield"))
Btn11 = Button(root, text="Greenwich", command=lambda: DisplayInfo("Greenwich"))
Btn12 = Button(root, text="Hackney", command=lambda: DisplayInfo("Hackney"))	
Btn13 = Button(root, text="Hammersmith\nand Fulham", command=lambda: DisplayInfo("Hammersmith and Fulham"))
Btn14 = Button(root, text="Haringey", command=lambda: DisplayInfo("Haringey"))
Btn15 = Button(root, text="Harrow", command=lambda: DisplayInfo("Harrow"))
Btn16 = Button(root, text="Havering", command=lambda: DisplayInfo("Havering"))
Btn17 = Button(root, text="Hillingdon", command=lambda: DisplayInfo("Hillingdon"))
Btn18 = Button(root, text="Hounslow", command=lambda: DisplayInfo("Hounslow"))
Btn19 = Button(root, text="Islington", command=lambda: DisplayInfo("Islington"))
Btn20 = Button(root, text="Kensington\nand Chelsea", command=lambda: DisplayInfo("Kensington and Chelsea"))
Btn21 = Button(root, text="Kingston\nupon\nThames", command=lambda: DisplayInfo("Kingston upon Thames"))
Btn22 = Button(root, text="Lambeth", command=lambda: DisplayInfo("Lambeth"))
Btn23 = Button(root, text="Lewisham", command=lambda: DisplayInfo("Lewisham"))
Btn24 = Button(root, text="Merton", command=lambda: DisplayInfo("Merton"))
Btn25 = Button(root, text="Newham", command=lambda: DisplayInfo("Newham"))	
Btn26 = Button(root, text="Redbridge", command=lambda: DisplayInfo("Redbridge"))
Btn27 = Button(root, text="Richmond\nupon Thames", command=lambda: DisplayInfo("Richmond upon Thames"))
Btn28 = Button(root, text="Southwark", command=lambda: DisplayInfo("Southwark"))
Btn29 = Button(root, text="Sutton", command=lambda: DisplayInfo("Sutton"))	
Btn30 = Button(root, text="Tower\nHamlets", command=lambda: DisplayInfo("Tower Hamlets"))
Btn31 = Button(root, text="Waltham\nForest", command=lambda: DisplayInfo("Waltham Forest"))	
Btn32 = Button(root, text="Wandsworth", command=lambda: DisplayInfo("Wandsworth"))
Btn33 = Button(root, text="Westminster", command=lambda: DisplayInfo("Westminster"))
Btn1.place(x=1060, y=360)
Btn2.place(x=620, y=200)
Btn3.place(x=1095, y=530)
Btn4.place(x=550, y=325)
Btn5.place(x=970, y=725)
Btn6.place(x=669, y=335)
Btn7.place(x=782, y=414)
Btn8.place(x=800, y=750)
Btn9.place(x=470, y=405)
Btn10.place(x=775, y=125)
Btn11.place(x=957, y=505)
Btn12.place(x=825, y=332)
Btn13.place(x=610, y=470)
Btn14.place(x=750, y=255)
Btn15.place(x=450, y=235)
Btn16.place(x=1210, y=320)
Btn17.place(x=285, y=360)
Btn18.place(x=410, y=500)
Btn19.place(x=755, y=350)
Btn20.place(x=590, y=425)
Btn21.place(x=520, y=650)
Btn22.place(x=730, y=525)
Btn23.place(x=870, y=550)
Btn24.place(x=635, y=635)
Btn25.place(x=940, y=390)	
Btn26.place(x=990, y=270)
Btn27.place(x=470, y=556)
Btn28.place(x=784, y=478)
Btn29.place(x=665, y=745)	
Btn30.place(x=840, y=394)
Btn31.place(x=873, y=235)
Btn32.place(x=625, y=548)
Btn33.place(x=680, y=424)

ClearButton = Button(root, text="Reset Positions", height=5, width=25, bg="#FFFFFF", fg="#000000", command=Clear)
ClearButton.place(x=1380, y=867)


NB = 1
Update()
    
#Sets the tkinter window 
root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.title("Rosey")
root.mainloop()

