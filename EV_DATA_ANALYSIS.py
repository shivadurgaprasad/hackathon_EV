#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
import pandas as pd
import plotly.express as px


# In[32]:


data  = pd.read_csv("C:\\Users\\Shiva\\Downloads\\dataset.csv")


# In[ ]:





# In[56]:


data


# In[ ]:


'''
The dataset  contains about Ev vehicles from the different data that is collected from 1995 to 2023 collected from diffrent country of  diffrent makers

County: The county where the vehicle is registered or located.

City: The city where the vehicle is registered or located.

State: The state where the vehicle is registered or located.

Postal Code: The postal code (ZIP code) where the vehicle is registered or located.

Model Year: The year of the vehicle's model.

Make: The make or manufacturer of the vehicle (e.g., Toyota, Honda, Ford, etc.).

model: The specific model name or designation of the vehicle (e.g., Camry, Accord, Mustang, etc.).

Electric Vehicle Type: The type of electric vehicle (e.g., battery electric vehicle - BEV, plug-in hybrid electric vehicle - PHEV).

Clean Alternative Fuel Vehicle (CAFV) Eligibility: Indicates whether the vehicle is eligible for Clean Alternative Fuel Vehicle incentives or programs.

Electric Range: The electric-only range of a plug-in hybrid electric vehicle (PHEV) or battery electric vehicle (BEV).

Base MSRP: The Manufacturer's Suggested Retail Price (MSRP) of the vehicle without any optional features.

Legislative District: The legislative district where the vehicle is registered or located.

DOL Vehicle ID: An identification number or code provided by the Department of Licensing (DOL) for the vehicle.

Vehicle Location: The specific location where the vehicle is located.

Electric Utility: The electric utility company that provides electricity to the area where the vehicle is located.

2020 Census Tract: A geographic area used for census purposes in the year 2020, which helps to identify the location of the vehicle.
'''


# In[57]:


data.info()


# In[58]:




# Checking t the number of duplicate rows in the entire DataFrame
duplicate_count = data.duplicated().sum()

print("Number of duplicate rows in the DataFrame:", duplicate_count)


# In[59]:


# Checking the missing values in the DataFrame
data.isnull().sum() 


# In[50]:


data.rename(columns={'Model': 'model'}, inplace=True)


# In[53]:


# Fill missing values in 'Model' column with the most common value

most_common_model = data['model'].mode().iloc[0]

data['model'].fillna(most_common_model, inplace=True)


# In[82]:


data['Legislative District'].isnull().sum()


# In[84]:


from sklearn.impute import KNNImputer


knn_imputer = KNNImputer(n_neighbors=5)  

numerical_columns = ['Legislative District']  
data['Legislative District'] = knn_imputer.fit_transform(data[numerical_columns])


# In[94]:


data['Legislative District'].isnull().sum()


# In[88]:


data[['Vehicle Location']]


# In[92]:


# filling the missing values in the vehicle Column
def extract_latitude_longitude(point):
   if pd.notnull(point):
       lat_long = point.strip('POINT ()').split()
       if len(lat_long) == 2:
           latitude = float(lat_long[1])
           longitude = float(lat_long[0])
           return latitude, longitude
   return None, None

# Applying the function to extract latitude and longitude from 'Vehicle Location'
data[['Latitude', 'Longitude']] = data['Vehicle Location'].apply(extract_latitude_longitude).apply(pd.Series)

# Defining a function to create the 'Vehicle Location' from 'City' and 'State'
def create_vehicle_location(row):
   if pd.notnull(row['City']) and pd.notnull(row['State']):
       return f"POINT ({row['Longitude']} {row['Latitude']})"
   else:
       return None

# Using 'City' and 'State' to fill missing values in 'Vehicle Location'
data['Vehicle Location'].fillna(data.apply(create_vehicle_location, axis=1), inplace=True)


# In[93]:


data['Vehicle Location'].isnull().sum()


# In[96]:


data['Electric Utility'].isnull().sum()


# In[103]:


import pandas as pd



# Filling missing 'Electric Utility' values with the mode of each group defined by 'County', 'City', and 'State'
data['Electric Utility'].fillna(data.groupby(['County', 'City', 'State'])['Electric Utility'].transform(lambda x: x.mode().iat[0] if not x.mode().empty else 'Unknown'), inplace=True)


# In[104]:


data['Electric Utility'].isnull().sum()


# In[109]:


data.drop(['Latitude','Longitude'], axis=1, inplace=True)


# In[323]:


data['Electric Range'] = data['Electric Range'].astype('int')


# In[322]:


data['Electric Range Category'].unique()


# In[329]:


import pandas as pd


grouped_data = data.groupby(['Make', 'model', 'Electric Vehicle Type'])['Electric Range Category'].apply(lambda x: x.mode().iloc[0] if not x.isnull().all() else None)

mapping_dict = grouped_data.to_dict()

data['Electric Range Category'] = data.apply(lambda row: mapping_dict.get((row['Make'], row['model'], row['Electric Vehicle Type']), row['Electric Range Category']) if pd.isnull(row['Electric Range Category']) else row['Electric Range Category'], axis=1)


# In[333]:


data


# In[337]:


import pandas as pd
from sklearn.impute import SimpleImputer


# Replace 'None' with NaN to represent missing values
data['Electric Vehicle Type'].replace('None', pd.NA, inplace=True)

# Create the SimpleImputer object with 'most_frequent' strategy
imputer = SimpleImputer(strategy='most_frequent')

# Fill the missing values in 'Electric Vehicle Type' column
data['Electric Vehicle Type'] = imputer.fit_transform(data[['Electric Vehicle Type']])


# In[336]:


data['Electric Range Category'].unique()


# In[338]:


# Data After Missing Values in the DataFrame
data.isnull().sum()


# In[ ]:


### EXPLORATORY DATA ANALYSIS


# In[111]:


data


# In[125]:



# Dictionary map state abbreviations to their full names
state_full_forms = {'FL': 'Florida','NV': 'Nevada','WA': 'Washington','IL': 'Illinois','NY': 'New York','VA': 'Virginia',                    'OK': 'Oklahoma','KS': 'Kansas','CA': 'California','NE': 'Nebraska','MD': 'Maryland','CO': 'Colorado','DC': 'District of Columbia',                    'TN': 'Tennessee','SC': 'South Carolina','CT': 'Connecticut','OR': 'Oregon','TX': 'Texas','SD': 'South Dakota',                    'HI': 'Hawaii','GA': 'Georgia','MS': 'Mississippi','AR': 'Arkansas','NC': 'North Carolina','MO': 'Missouri',                    'UT': 'Utah','PA': 'Pennsylvania','DE': 'Delaware','OH': 'Ohio','WY': 'Wyoming','AL': 'Alabama',                    'ID': 'Idaho','AZ': 'Arizona','AK': 'Alaska','LA': 'Louisiana','NM': 'New Mexico','WI': 'Wisconsin',                    'KY': 'Kentucky','NJ': 'New Jersey','MN': 'Minnesota','MA': 'Massachusetts','ME': 'Maine','RI': 'Rhode Island',                    'NH': 'New Hampshire','ND': 'North Dakota'}

data['State'] = data['State'].map(state_full_forms)



# In[126]:


data


# In[133]:


# Univarite Analysis

import plotly.express as px

state_counts = data['State'].value_counts()
fig = px.bar(x=state_counts.index, y=state_counts.values, labels={'x': 'State', 'y': 'Count'})
fig.update_layout(title_text='State-wise Vehicle Counts')
fig.show()


# In[137]:


#observations
'''
1) The state of Washington stands out with the highest number of registered vehicles among all the states in the dataset. and dataset also more baised to the washington

 2)This suggests that Washington residents show a strong preference for owning vehicles compared to residents of other states.
 
 3)  This might Due to growth economy and job oppurtunites that encourage people may lead to a higher demand for personal vehicles.
'''


# In[139]:


fig = px.histogram(data, x='Electric Range', nbins=20, labels={'Electric Range': 'Electric Range'})
fig.update_layout(title_text='Distribution of Electric Range')
fig.show()


# In[142]:


cafv_counts = data['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts()
fig = px.pie(data, names=cafv_counts.index, values=cafv_counts.values, title='CAFV Eligibility')
fig.show()


# In[ ]:


# observations
'''
1) from above pie chart Around 52.1% of vehicles in the dataset are eligible for Clean Alternative Fuel Vehicle (CAFV) status. 
This indicates a notable proportion of environmentally friendly vehicles, suggesting a positive trend in adopting cleaner transportation options.

2) there is Approximately 32.1% of vehicles have unknown eligibility as their battery range has not been researched. 
This suggests a lack of comprehensive data or readily available information on these vehicles. Further research or data collection is necessary to determine their eligibility for clean alternative fuel vehicle status.

3)Around 13.1% of vehicles are ineligible for clean alternative fuel status due to their low battery range. 
This highlights the need for improvements in battery technology and the adoption of cleaner technologies to enhance the proportion of environmentally friendly vehicles in the population.
'''


# In[150]:


ev_type_counts = data['Electric Vehicle Type'].value_counts()
fig = px.pie(data, names=ev_type_counts.index, values=ev_type_counts.values, title='Electric Vehicle Types')
fig.show()


# In[ ]:


'''Total number of Electric Vehicle Types: 112634
The most common Electric Vehicle Type is 'Battery Electric Vehicle (BEV)' with 86044 occurrences.
It accounts for approximately 76.39% of the total Electric Vehicle Types.'''


# In[153]:


import plotly.express as px

ev_type_counts = data['Electric Vehicle Type'].value_counts()

fig = px.bar(data, y=ev_type_counts.index, x=ev_type_counts.values, 
             title='Electric Vehicle Types', orientation='h', labels={'x': 'Count', 'y': 'EV Type'})
fig.show()


# In[ ]:


'''
1) The above bar chart shows that clear dominance of Battery Electric Vehicles (BEVs) in the dataset, with an overwhelming count of 86,044 occurrences.
BEVs constitute approximately 76.39% of the total electric vehicle types, establishing them as the most favourite among the vehicles in the dataset.

2)The substantial presence of Battery Electric Vehicles (BEVs) in the dataset indicates a notable shift towards all-electric vehicles, surpassing hybrid and internal combustion engine vehicles.
This trend is likely fueled by rising environmental awareness and advancements in battery technology, making BEVs the preferred choice among consumers and fleet operators.


'''


# In[156]:


top_manufacturers = data['Make'].value_counts().nlargest(10)
fig = px.bar(x=top_manufacturers.index, y=top_manufacturers.values,
             labels={'x': 'Manufacturer', 'y': 'Vehicle Count'})
fig.update_layout(title_text='Top 10 Manufacturers by Vehicle Count')
fig.show()


# In[ ]:


'''
Observations

1)The bar chart highlights the top 10 manufacturers based on vehicle count.
clearly indicating significant variations among them

2)Tesla as the undisputed leader among the top 10 manufacturers in terms of vehicle count.
With over 50,000 vehicles, Tesla has a substantial lead over its competitors.

3)While Tesla has a massive lead, Nissan stands out as a strong player in the market, securing the second position. 

4)The high vehicle count for certain  Tesla manufacturers suggests  that  Tesla created strong brand loyalty among consumers.
They have established a positive reputation and trust among their customers
'''


# In[169]:


fig = px.histogram(data, x='Model Year', nbins=10, labels={'Model Year': 'Model Year'})
fig.update_layout(title_text='Distribution of Model Year')
fig.show()


# In[ ]:


'''
1) The histogram indicates limited or no electric vehicle entries in the dataset for 1995 to 2005,
reflecting the scarcity of commercially available EVs during that period. 
This aligns with historical context, as EVs were in early stages of development with less mature technology compared to today.

2)The histogram starts to show an increase in the number of EV registrations starting from the year 2010.
This significant surge suggests that around this time, electric vehicles started to experience a substantial growth phase in the automotive market. 
The rise in the number of EVs indicates a growing adoption and popularity of electric vehicles among consumers

3) 2015 to 2020 - Rapid Growth Period: The histogram suggests a significant surge in the number of electric vehicles from 2015 to 2020.
This period marks a phase of rapid growth for electric vehicles

4) Post-2020 - Potential Continued Growth: While the histogram does not show data beyond 2020, 
it's reasonable to expect that the trend of increasing electric vehicle adoption continued in the years following
'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Bivariate Analysis


# In[218]:


import pandas as pd
import plotly.express as px


top_models = data.groupby('Make')['model'].value_counts().reset_index(name='Count')
top_models = top_models.sort_values(by=['Make', 'Count'], ascending=[True, False])
top_models = top_models.groupby('Make').head(1)

# Step 2: Create the plot using Plotly bar plot
fig = px.bar(top_models, x='Make', y='Count', color='model', title='Top Sold Model for Each Make')
fig.update_layout(xaxis_title='Make', yaxis_title='Count')
fig.show()


# In[ ]:


'''
1) The bar plot showcases the most popular or top-selling models for each vehicle make.
The height of each bar represents the number of units sold for that particular model within its respective make

2) "Tesla Model 3," indicating it is the highest selling model with an impressive count of 23,000 units. This remarkable sales figure highlights the overwhelming popularity of the Tesla Model 3 among customers,
making it the most preferred choice in the market.

3)he Nissan "Leaf" stands out as the second-highest selling model, boasting substantial sales of 12,880 units. 
These insights reflect the strong market demand for electric vehicles, with both Tesla Model 3 and Nissan Leaf emerging as top contenders in the highly competitive electric vehicle market.
'''


# In[ ]:





# In[340]:


import plotly.express as px

maker_count_by_year = data.groupby(['Model Year', 'Make']).size().reset_index(name='Count')

maker_count_by_year = maker_count_by_year.sort_values(by=['Model Year', 'Count'], ascending=[True, False])

top_makers_by_year = maker_count_by_year.groupby('Model Year').first().reset_index()

fig = px.bar(top_makers_by_year, x='Model Year', y='Count', text='Make',
             title="Top Car Makers Over the Years",
             labels={'Count': 'Number of Cars', 'Model Year': 'Year'},
             template='plotly_white')

# Rotate x-axis labels for better visibility
fig.update_layout(xaxis_tickangle=-45)

# Show the plot
fig.show()


# In[ ]:


'''
observations :



-> 1997 to 2010: There was no significant growth in the EV market, even though prominent car makers like Chevrolet, Toyota, Ford, and Tesla were present.

-> 2011: The EV market started to show slight growth, and this is when Tesla entered the market in 2008. In 2011, Nissan's sales stood out compared to other brands.

-> 2016: Tesla experienced a notable increase in sales, solidifying its position in the EV market.

-> 2017: Chevrolet saw a surge in sales, making a notable impact.

-> 2018: Tesla's sales increased again, further establishing its dominance in the EV market.
 
-> 2019: Tesla experienced a slight decrease in sales, but still remained a significant player in the industry.

-> 2020 to 2022: Tesla's sales began to increase again, indicating sustained growth in EV vehicle sales.
'''


# In[343]:


data['Electric Range']=data['Electric Range'].astype('int')


# In[346]:


import plotly.express as px

# Group the data by Model Year and Electric Vehicle Type to get the count of each type for each year
ev_type_count_by_year = data.groupby(['Model Year', 'Electric Vehicle Type']).size().reset_index(name='Count')

# Create the line plot with markers
fig = px.line(ev_type_count_by_year, x='Model Year', y='Count', color='Electric Vehicle Type', markers=True,
              title="Electric Vehicle Type Preference Over the Years",
              labels={'Count': 'Number of Cars', 'Model Year': 'Year'},
              template='plotly_white')

# Show the plot
fig.show()


# In[ ]:


'''
From the above graph

1) there is battery electric vehicle  sales from the started from 1997 year to 2010 there is constant in sales .

2) when coming to 2011 the plug in hybrid electric electic vechiles came to market

3) as years coming to BEV vechiles become more popular due to environmental friendly and the rise in fuel prices
'''


# In[ ]:





# In[347]:


data


# In[348]:


import plotly.express as px

# Group the data by Make and Electric Range Category to get the count of each category for each make
make_range_count = data.groupby(['Make', 'Electric Range Category']).size().reset_index(name='Count')

# Create the grouped bar plot
fig = px.bar(make_range_count, x='Make', y='Count', color='Electric Range Category',
             title="Electric Range Category by Car Make",
             labels={'Count': 'Number of Cars', 'Make': 'Car Make'},
             template='plotly_white')

# Show the plot
fig.show()


# In[ ]:



'''

obeservations:

1) we can majority cars which have high Range were produced bt Tesla when compared to other brands

2)compare to tesla most of the other brands were low range cars they were producing

3) cheverlot has few cars that has high range
'''


# In[349]:


data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Task 3 : Create a Choropleth to display the number of EV vehicles based on location.


# In[267]:


import pandas as pd
import plotly.express as px



# Performing the groupby operation to calculate EV count for each location
ev_location_count = data.groupby(['City', 'State']).size().reset_index(name='EV Count')

#  Creating the Choropleth map
fig = px.choropleth(
    ev_location_count,
    locations='State',                # Column with the state abbreviations (e.g., 'FL', 'NV', 'WA')
    locationmode='USA-states',        # Use USA states as the location mode
    color='EV Count',                 # Value to be color-mapped (Number of EV vehicles)
    scope="usa",                      # Set the scope to display only the USA map
    hover_name='City',                # Column with city names for hover information
    title='Number of EV Vehicles by Location',
    color_continuous_scale=px.colors.sequential.Plasma,  # Choose the color scale
)

#  layout for visualization
fig.update_layout(
    geo=dict(
        showcoastlines=True,          # Show coastlines on the map
        coastlinecolor="DarkGrey",    # Set the color for coastlines (dark grey)
        showland=True,                # Show land areas
        landcolor="Black",            # Set the color for land areas (black)
        showlakes=True,               # Show lakes on the map
        lakecolor="LightBlue",            # Set the color for lakes (black)
    ),
)

#  Showing the Choropleth map
fig.show()


# In[ ]:


'''
1)The Choropleth map showcases the variation in EV adoption across different states. States with a higher density of color indicate a larger number of EV vehicles,
suggesting higher adoption rates.

2) The States with a significant number of EV vehicles demonstrate a supportive environment for electric mobility.
These states may offer incentives, tax benefits, or a robust charging infrastructure, fostering a favorable climate for EV adoption.

3) High EV counts in certain cities and states may have a positive economic impact, 
driving employment opportunities in EV-related industries like manufacturing, sales, and service.


'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Task 3: Create a Racing Bar Plot to display the animation of EV Make and its count each year.


# In[275]:


import pandas as pd
import plotly.express as px


data['Model Year'] = pd.to_datetime(data['Model Year'], format='%Y').dt.year.astype(str)

# Grouping data by 'Model Year' and 'Make' to get the count of each EV Make for each year
ev_make_count = data.groupby(['Model Year', 'Make']).size().reset_index(name='Count')

# Creating  the Racing Bar Plot
fig = px.bar(
    ev_make_count,
    x='Make',
    y='Count',
    animation_frame='Model Year',
    animation_group='Make',
    orientation='v',                                     # Set orientation to vertical
    range_y=[0, ev_make_count['Count'].max() + 10],      # Adjust y-axis range for better visualization
    title='EV Make Count Over the Years',
    color_discrete_sequence=px.colors.qualitative.Bold,  # Use a different color palette (Bold)
    labels={'Make': 'EV Make', 'Count': 'Count'},        # Customize axis labels
)

# creating  the layout for better visualization
fig.update_layout(
    xaxis_title='EV Make',
    yaxis_title='Count',
    showlegend=False,       # Remove the legend
    title_font_size=24,
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    bargap=0.1,            # Reduce the gap between bars for a compact look
    bargroupgap=0.2,       # Adjust the gap between bar groups for a better layout
)

# Show the Racing Bar Plot
fig.show()


# In[ ]:


'''
1)The Racing Bar Plot reveals a compelling trend in the electric vehicle market - the consistent increase in the number of EV makes over the years.

2)This surge implies a growing diversity in the market, with numerous new manufacturers and brands entering the electric vehicle space.

3)This pattern showcases the industry's continuous evolution, marked by innovation and competition as companies strive to carve their place in the fast-paced world of electric mobility

4) The dynamic movement of bars reflects the changing market share of different EV makes over time.
Some EV manufacturers may have experienced significant growth in popularity, leading to higher counts, while others may have faced fluctuations or even declining counts.



'''

