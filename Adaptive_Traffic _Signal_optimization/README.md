# 🚦 Adaptive Traffic Signal Optimization Using Deep Learning for Smarter Traffic Management  

This project demonstrates a **smart traffic management system** that integrates advanced **deep learning (YOLOv8)** for real-time vehicle detection and a **genetic algorithm** for adaptive signal timing optimization. The solution is designed to reduce congestion, improve traffic flow, and minimize environmental impacts.  

---

## **📌 Problem Statement**  
Traditional traffic signals operate on fixed cycles, often failing to adapt to real-time traffic conditions, resulting in:  
- 🚗 **Long wait times** at intersections  
- 🛣️ **Inefficient traffic flow** during peak hours  
- 🌍 **Increased fuel consumption and emissions**  

---

## **⚙️ Solution Overview**  
1. **🔍 Vehicle Detection (YOLOv8)**  
   - Leveraged the YOLOv8 model to detect and count vehicles in real time from traffic camera feeds.  
   - Accurate identification of vehicle density for each direction at an intersection.  

2. **🧬 Traffic Optimization (Genetic Algorithm)**  
   - Applied a genetic algorithm to dynamically allocate green light durations based on vehicle count.  
   - The algorithm evolves optimal signal timings to balance traffic flow across all lanes.  

---

## **🛠️ Tools and Technologies Used**  
- **Deep Learning**: YOLOv8 for real-time object detection  
- **Optimization Algorithm**: Genetic Algorithm for adaptive signal timing  
- **Programming**: Python for implementation  
- **Libraries**:  OpenCV, NumPy, Matplotlib  

---

## **📊 Workflow**  
1. **Data Collection**:  
   - Captured traffic images and videos from an urban intersection.  

2. **Model Training and Testing**:  
   - Fine-tuned YOLOv8 for vehicle detection tasks.  

3. **Integration**:  
   - Real-time vehicle detection outputs fed into the genetic algorithm.  

4. **Optimization**:  
   - The genetic algorithm determines signal timings to minimize queue lengths and maximize flow.  

---

## **🔑 Key Features**  
- **Real-Time Adaptation**: Automatically adjusts signal timings based on live traffic conditions.  
- **Efficiency**: Reduces vehicle wait times and optimizes flow for all directions.  
- **Scalability**: Easily deployable at multiple intersections.  

---

## **🌟 Project Details**  
- **Project Title**: Adaptive Traffic Signal Optimization Using Deep Learning for Smarter Traffic Management  
- **Semester**: 7  
- **Team Members**:  
  - Zurin Lakdawala  
  - Mit Thaker  
- **Guide**: Prof. Hardik Soni  

---

## **📈 Results**  
- Achieved a significant reduction in average vehicle wait time at intersections.  
- Improved traffic throughput and reduced congestion compared to fixed-cycle signals.  

---

## **🚀 Future Scope**  
- Incorporate pedestrian detection for better safety.  
- Expand to include anomaly detection for incidents like accidents or breakdowns.  
- Integrate IoT devices for a more connected traffic management system.  

## **🎥 Demo Video**

To better understand our system's workflow and results, watch the demo video below:

[Watch the Demo Video](https://github.com/Zurinlakdawala91/Adaptive-Traffic-Signal-Optimization-Using-Deep-Learning-for-Smarter-Traffic-Management/blob/main/Screen%20Recording%202024-12-03%20112151.mp4)



---  
🎯 *This project marks a step forward in smarter traffic management, providing a scalable and efficient solution to modern urban challenges.*  
