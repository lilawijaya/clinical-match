import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score

def find_top_hubs(ideal_age, ideal_gender, ideal_race, ideal_education, ideal_income):
    # Create a synthetic dataset for x community hubs in Oakland
    center_names = ['Alameda County Community Food Bank', 'Allen Temple Baptist Church', 'African American Museum and Library at Oakland', 'Asian Health Services', 'Berkeley Public Library', 
                    'California College of the Arts', 'Chabot Space and Science Center', 'Children’s Fairyland', 'Crab Cove Visitor Center', 'East Bay Meditation Center',
                    'East Oakland Sports Center', 'Fruitvale Transit Village', 'Grand Lake Theater', 'Harbor Bay Club', 'Jack London Square', 'Lake Merritt', 
                    'Lakeside Park Garden Center', 'Laney College', 'Mills College', 'Montclair Branch Library', 'Morcom Rose Garden', 'Mosswood Park',
                    'Oakland Art Murmur', 'Oakland City Hall', 'Oakland Museum of California', 'Oakland Public Library - Main Library', 'Oakland Zoo', 'Preservation Park', 
                    'Pro Arts', 'Rockridge Market Hall', 'Rotary Nature Center at Lake Merritt', 'Ruth Bancroft Garden', 'SPLASH Pad Park', 'Sausal Creek Outfall Restoration',
                    'Temescal Branch Library', 'The Cathedral of Christ the Light', 'The Crucible', 'The Flight Deck', 'The New Parkway Theater', 'The Uptown Nightclub',
                    'The West Oakland Youth Center', 'The Women’s Building', 'USS Hornet Museum', 'Unity Council', 'Warriors Ground SF', 'Yoshi’s Oakland', 'Zoo Labs']
    x = len(center_names)
    df = pd.DataFrame({
        'hub_id': range(1, x+1),
        'center_name': center_names,
        'age': np.random.randint(18, 85, size=x),
        'gender': np.random.choice(['Male', 'Female', 'Non-binary'], size=x),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], size=x),
        'education': np.random.choice(['Less than high school', 'High school', 'Some college', 'Bachelor', 'Graduate'], size=x),
        'income': np.random.randint(10000, 120000, size=x)
    })

    # One-hot encode categorical features
    cat_cols = ['gender', 'race', 'education']
    enc = OneHotEncoder()
    encoded_cats = enc.fit_transform(df[cat_cols])
    encoded_cat_df = pd.DataFrame(encoded_cats.toarray(), columns=enc.get_feature_names(cat_cols))
    df = pd.concat([df, encoded_cat_df], axis=1)

    # Normalize the numeric features
    numeric_cols = ['age', 'income']
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    # Assign weights to each criteria
    weights = {'age': 0.2, 'gender_Female': 0.1, 'gender_Male': 0.1, 'gender_Non-binary': 0.1, 'race_Asian': 0.1, 'race_Black': 0.1, 'race_Hispanic': 0.1, 'race_Other': 0.1, 'race_White': 0.1, 'education_Bachelor': 0.15, 'education_Graduate': 0.15, 'education_High school': 0.05, 'education_Less than high school': 0.05, 'education_Some college': 0.05, 'income': 0.1}

        # Use KMeans to cluster the data into 5 groups
    kmeans = KMeans(n_clusters=5)
    X = df.drop(['hub_id', 'center_name', 'gender', 'race', 'education'], axis=1)
    kmeans.fit(X)
    df['cluster'] = kmeans.predict(X)

    # Compute the silhouette score for the clustering
    silhouette_avg = silhouette_score(X, kmeans.labels_)

    # Compute a score for each hub based on the ideal criteria
    df['score'] = (weights['age'] * abs(df['age'] - ideal_age) +
                weights['gender_Female'] * (df['gender_Female'] != ideal_gender) +
                weights['gender_Male'] * (df['gender_Male'] != ideal_gender) +
                weights['gender_Non-binary'] * (df['gender_Non-binary'] != ideal_gender) +
                weights['race_Asian'] * (df['race_Asian'] != ideal_race) +
                weights['race_Black'] * (df['race_Black'] != ideal_race) +
                weights['race_Hispanic'] * (df['race_Hispanic'] != ideal_race) +
                weights['race_Other'] * (df['race_Other'] != ideal_race) +
                weights['race_White'] * (df['race_White'] != ideal_race) +
                weights['education_Bachelor'] * (df['education_Bachelor'] != ideal_education) +
                weights['education_Graduate'] * (df['education_Graduate'] != ideal_education) +
                weights['education_High school'] * (df['education_High school'] != ideal_education) +
                weights['education_Less than high school'] * (df['education_Less than high school'] != ideal_education) +
                weights['education_Some college'] * (df['education_Some college'] != ideal_education) +
                weights['income'] * abs(df['income'] - ideal_income))


    # Rank the hubs based on their score and return the top 5
    top_hubs = df.sort_values('score', ascending=False).head(5)


    # Implement machine learning mechanism to make smarter decisions
    # Create a machine learning model to predict the score for each hub
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, df['score'], test_size=0.2, random_state=42)

    # Train the random forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Use the model to predict the score for each hub
    df['predicted_score'] = rf.predict(X)

    # Compute the correlation between the predicted and actual scores
    corr = np.corrcoef(df['score'], df['predicted_score'])[0,1]
    print('Correlation between predicted and actual scores:', corr)

    return top_hubs
