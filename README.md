<h1>Virtual Guru<h1>

<h2>Phase 1: Using Random Forest</h2>
    <p>For the initial phase of the project, a Random Forest Classifier was employed.</p>
    <ul>
        <li><strong>Dataset:</strong> Due to unavailability of proper dataset, generated using the script in <a href = "https://github.com/parikhaitan/Virtual-Guru/blob/main/virtual%20guru-%20random%20forest.ipynb">virtual-guru random forest.ipynb.</a></li>
        <li><strong>Accuracy:</strong> 99.28%</li>
    </ul>

  <h3>Advantages</h3>
    <ul>
        <li>Utilizes an ensemble learning technique that can handle complex relationships in the data.</li>
        <li>Less prone to overfitting compared to individual decision trees.</li>
        <li>Provides feature importances, aiding in understanding the most influential factors.</li>
    </ul>

  <h3>Limitations</h3>
    <ul>
        <li>Cannort adapt to changes over time to improve prediction accuracy with feedback.</li>
    </ul>


<h2>Phase 2: Adaptive Learner (Using TensorFlow)</h2>

 <h3>Model Architecture</h3>
    <ul>
        <li><strong>Input Layer:</strong> Receives various input features including learner interactions, performance data, content-based features, and optional demographic information.</li>
        <li><strong>Hidden Layers:</strong> Multiple layers applying non-linear transformations (ReLU activation) to capture complex patterns in the data.</li>
        <li><strong>Output Layer:</strong> Produces a single value representing the predicted learner performance on future assessments.</li>
    </ul>

  <h3>Online Learning with Feedback</h3>
    <ul>
        <li><strong>Feedback Collection:</strong> Collects both explicit (direct user input) and implicit (user interactions) feedback.</li>
        <li><strong>Feedback Integration:</strong> Incorporates feedback into the loss function during training to improve prediction accuracy over time.</li>
    </ul>

  <h3>Advantages</h3>
    <ul>
        <li><strong>Personalized Learning:</strong> Tailors the learning experience to individual needs and preferences.</li>
        <li><strong>Continuous Improvement:</strong> Adapts to changes over time and improves prediction accuracy with feedback.</li>
        <li><strong>Scalability:</strong> Capable of handling large datasets and user volumes for real-world applications.</li>
    </ul>

  <h3>Limitations</h3>
    <ul>
        <li><strong>Data Requirements:</strong> Requires substantial data for effective training.</li>
    </ul>

  
