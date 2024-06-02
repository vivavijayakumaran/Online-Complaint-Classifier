#ONLINE REVIEW CLASSIFICATION
#The assignment is to create a tool that trains several machine learning models to perform the 
#task of classifying online reviews. Some of these online reviews refer to hazardous products, so 
#these machine learning models will help to identify the most serious product complaints. 

import nltk, math, google.colab.files,textblob, json, joblib, matplotlib.pyplot as plt, requests, sklearn.neighbors, sklearn.neural_network, sklearn.metrics, sklearn.model_selection, sklearn.tree
nltk.download("all",quiet=True)
import warnings 
import time
warnings.filterwarnings("ignore")
start_time = time.time() 
response = requests.get("https://dgoldberg.sdsu.edu/515/appliance_reviews.json")
if response:
  data = json.loads(response.text)
  all_reviews = ""
  all_words = []
  all_reviews_list=[]
  relevant_words=[]
  y=[]
  for line in data:
    review = line["Review"]
    review=review.lower()
    #print(review)
    stars = line["Stars"]
    safety_hazard = line["Safety hazard"]
    #all_reviews.append(review)
    all_reviews=all_reviews+review+""
    all_reviews_list.append(review)
    y.append(safety_hazard)
  end_time = time.time() # Timestamp for when process ended 
  time_elapsed = end_time - start_time # Difference between times 
  print("\nLoading data...")
  print("Completed in", time_elapsed, "seconds.")

  start_time = time.time() 
  blob = textblob.TextBlob(all_reviews)
  for word in blob.words:
    words=blob.words
    all_words.append(word)
  #print("All words:",len(all_words))
  unique_list=[]
  for word in all_words:
    if word not in unique_list:
      unique_list.append(word)
  
  end_time = time.time() 
   
  time_elapsed = end_time - start_time 
  print("\nIdentifying unique words...")
  print("Completed in", time_elapsed, "seconds.")

  start_time = time.time() 
  for word in unique_list:
    A=0
    B=0
    C=0
    D=0
    relevance_scores=[]
    for line in data:
      review = line["Review"]
      review=review.lower()
      safety_hazard = line["Safety hazard"]
      if word in review and safety_hazard==1:
        A+=1
      elif word in review and safety_hazard==0:
        B+=1
      elif word not in review and safety_hazard==1:
        C+=1
      elif word not in review and safety_hazard==0:
        D+=1
  
    try:
      
      relevance_score=(((A+B+C+D)**(1/2))*((A*D)-(C*B)))/(((A+B)*(C+D))**(1/2))
      
      
    except:
      relevance_score=0
    
    if relevance_score >=4000:
      relevance_scores.append(relevance_score)
      relevant_words.append(word)
      
      start_time = time.time() 

      x=[]
      for review in all_reviews_list:
        z=[]
        for rel_word in relevant_words:
          if rel_word in review:
            z.append("1")
          else:
            z.append("0")
        x.append(z)
  end_time = time.time() 
  time_elapsed = end_time - start_time 
  print("\nGenerating relevance scores...")
  print("Completed in", time_elapsed, "seconds.")
  end_time = time.time() 
  time_elapsed = end_time - start_time  
  print("\nFormatting 2D list... ")
  print("Completed in", time_elapsed, "seconds.")

  start_time = time.time()
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y)

  # Decision tree
  dt_clf = sklearn.tree.DecisionTreeClassifier()
  dt_clf = dt_clf.fit(x_train, y_train)
  dt_predictions = dt_clf.predict(x_test)
  dt_accuracy = sklearn.metrics.accuracy_score(y_test, dt_predictions)
  

  # KNN
  knn_clf = sklearn.neighbors.KNeighborsClassifier(5)
  knn_clf = knn_clf.fit(x_train, y_train)
  knn_predictions = knn_clf.predict(x_test)
  knn_accuracy = sklearn.metrics.accuracy_score(y_test, knn_predictions)
  

  # Neural network
  nn_clf = sklearn.neural_network.MLPClassifier()
  nn_clf = nn_clf.fit(x_train, y_train)
  nn_predictions = nn_clf.predict(x_test)
  nn_accuracy = sklearn.metrics.accuracy_score(y_test, nn_predictions)
  
  end_time = time.time() # Timestamp for when process ended 
  time_elapsed = end_time - start_time # Difference between times 
  print("\nTraining machine learning models... ")
  print("Completed in", time_elapsed, "seconds.")
  print("\nDecision tree accuracy :", dt_accuracy)
  print("k-nearest neighbors accuracy:", knn_accuracy)
  print("Neural network accuracy:", nn_accuracy)

  if(dt_accuracy>knn_accuracy and dt_accuracy>nn_accuracy):
    highest_accuracy="Decision tree model"
  elif(knn_accuracy>dt_accuracy and knn_accuracy>nn_accuracy):
    highest_accuracy="k-nearest neighbors model"
  elif(nn_accuracy>dt_accuracy and nn_accuracy>knn_accuracy):
    highest_accuracy="Neural network model"
  
  #Export decision tree model using joblib
  joblib.dump(nn_clf, "model.joblib")
  google.colab.files.download("model.joblib")
  print(highest_accuracy, "performed best; saved to model.joblib.")

else:
  print("Sorry, Connection error.")
