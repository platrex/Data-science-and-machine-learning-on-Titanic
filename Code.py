{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "48dacb4f-7b46-4014-9874-db483d665285",
   "metadata": {
    "panel-layout": {
     "height": 184.1875,
     "visible": true,
     "width": 100
    }
   },
   "source": [
    "# Data Science Project\n",
    "## Titanic Dataset\n",
    "## ***Team members***:\n",
    "\n",
    "- ***Hussein Hodroj***\n",
    "- ***Hamza Abdelhadi***\n",
    "- ***Seedre Alyamani***\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e114c0de",
   "metadata": {
    "panel-layout": {
     "height": 0,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "import warnings \n",
    "import numpy as np \n",
    "import pandas as pd \n",
    "import matplotlib.pyplot as plt \n",
    "import seaborn as sns \n",
    "plt.style.use('fivethirtyeight') \n",
    "get_ipython().run_line_magic('matplotlib', 'inline')\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ba8f0b58",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set file paths for Titanic dataset\n",
    "file_path = r'C:\\Users\\ONE BY ONE\\Desktop\\Data_science titanic project\\titanic\\train.csv'\n",
    "\n",
    "file_path2=r'C:\\Users\\ONE BY ONE\\Desktop\\Data_science titanic project\\titanic\\test.csv'\n",
    "#read training dataset into a data frame\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "#read testing dataset into a data frame\n",
    "dft=pd.read_csv(file_path2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8e227307",
   "metadata": {
    "panel-layout": {
     "height": 433.20001220703125,
     "visible": true,
     "width": 100
    },
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "df.head(15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b23e10e9-64b5-46ac-8e9c-1992f4b67d0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "dft.head(15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5fe0df74",
   "metadata": {
    "panel-layout": {
     "height": 0,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "df.info() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98f3f85b",
   "metadata": {
    "panel-layout": {
     "height": 354,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "df.isnull().sum() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab648359",
   "metadata": {
    "panel-layout": {
     "height": 248.39999389648438,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "168e91f9-f192-4321-9bc0-1eee9997d772",
   "metadata": {},
   "source": [
    "--------------------------------------------------------------------------------\n",
    "### Counting Embarkation Points\n",
    "- **Southampton (S)**: Count of passengers who embarked at Southampton."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6d8c3791",
   "metadata": {
    "panel-layout": {
     "height": 0,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "count_S = (df['Embarked'] == 'S').sum()\n",
    "\n",
    "print(\"Number of occurrences of 'S' in the 'Embarked' column:\", count_S)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2a3043b-acee-401c-9708-0313350481d2",
   "metadata": {},
   "source": [
    "---------------------------------------------------------------------\n",
    "- **Cherbourg (C)**: Count of passengers who embarked at Cherbourg"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd6a6465",
   "metadata": {
    "panel-layout": {
     "height": 0,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "count_C = (df['Embarked'] == 'C').sum()\n",
    "\n",
    "print(\"Number of occurrences of 'C' in the 'Embarked' column:\", count_C)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f477cf86-cf0b-45d9-a3b7-0940c521bbca",
   "metadata": {},
   "source": [
    "---------------------------------------------------------------------\n",
    "- **Queenstown (Q)**: Count of passengers who embarked at Queenstown."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "203f9537",
   "metadata": {
    "panel-layout": {
     "height": 0,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "count_Q = (df['Embarked'] == 'Q').sum()\n",
    "\n",
    "print(\"Number of occurrences of 'Q' in the 'Embarked' column:\", count_Q)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bffc668f",
   "metadata": {
    "panel-layout": {
     "height": 0,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "missing_values_embarked = df['Embarked'].isnull().sum()\n",
    "\n",
    "print(\"Number of missing values in the 'Embarked' column:\", missing_values_embarked)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d1696093-064b-4431-8ed5-1b2bfcd5cd43",
   "metadata": {},
   "source": [
    "------------------------------\n",
    "## Filling Missing Embarked Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eff1b09c",
   "metadata": {
    "panel-layout": {
     "height": 0,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "# Fills missing values in 'Embarked' with 'S' (Southampton).\n",
    "df['Embarked'].fillna('S', inplace=True)\n",
    "\n",
    "count_S = (df['Embarked'] == 'S').sum()\n",
    "\n",
    "print(\"Number of occurrences of 'S' in the 'Embarked' column:\", count_S)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92befd5a",
   "metadata": {
    "panel-layout": {
     "height": 0,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "missing_values_embarked = df['Embarked'].isnull().sum()\n",
    "\n",
    "print(\"Number of missing values in the 'Embarked' column:\", missing_values_embarked)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff96dde7-45f2-443e-9193-8767d26da1fe",
   "metadata": {},
   "source": [
    "# -----visualization-----\n",
    "## Titanic Death Pie Chart\n",
    " - **Outcome Comparison**: Died vs. Survived."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5f17c8ae-a97b-4b5b-b256-f92a8383436c",
   "metadata": {
    "panel-layout": {
     "height": 874,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Create a figure and axis for the pie chart\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "\n",
    "#Define the colors\n",
    "colors = ['red', 'white']\n",
    "\n",
    "#Plot the pie chart with colors\n",
    "df['Survived'].value_counts().plot.pie(\n",
    "    explode=[0, 0.1], \n",
    "    autopct='%1.1f%%', \n",
    "    ax=ax, \n",
    "    shadow=True, \n",
    "    colors=colors,\n",
    "    startangle=140,\n",
    "    wedgeprops={'edgecolor': 'black', 'linewidth': 3}\n",
    ")\n",
    "\n",
    "# Set the title and remove the y-axis label\n",
    "ax.set_title('Survivors and the Death percentage', fontsize=16, fontweight='bold')\n",
    "ax.set_ylabel('')\n",
    "\n",
    "#Add a legend\n",
    "ax.legend(['Dead (0)', 'Survived (1)'], loc='upper right', fontsize=14, frameon=False)\n",
    "\n",
    "ax.axis('equal')\n",
    "\n",
    "# Show the plot\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a214af19-a698-4f40-bb7e-a842b2ec052e",
   "metadata": {},
   "source": [
    "-----\n",
    "## Titanic Death Bar Chart \n",
    "- **Gender Distribution**: Shows death count."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0d3b6d01",
   "metadata": {
    "panel-layout": {
     "height": 874,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "not_survived_df = df[df['Survived'] == 0]\n",
    "\n",
    "\n",
    "death_count_by_gender = not_survived_df['Sex'].value_counts()\n",
    "\n",
    "\n",
    "sns.set(style=\"whitegrid\")\n",
    "\n",
    "colors = ['skyblue', 'lightcoral']  # Blue for Male and Orange for Female\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size\n",
    "bars = death_count_by_gender.plot(kind='bar', color=colors, ax=ax, width=0.6)  # Custom color palette and bar width\n",
    "\n",
    "ax.set_title('Death Count by Gender on the Titanic', fontsize=18, fontweight='bold')  # Increase title font size and set bold\n",
    "ax.set_xlabel('Gender', fontsize=15, fontweight='bold')  # Increase x-axis label font size and set bold\n",
    "ax.set_ylabel('Death Count', fontsize=15, fontweight='bold')  # Increase y-axis label font size and set bold\n",
    "\n",
    "ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)  # Increase tick label font size and style\n",
    "ax.set_xticklabels(['Male', 'Female'], rotation=0, fontweight='bold')  # Set custom x-axis tick labels and set bold\n",
    "\n",
    "for bar in bars.patches:\n",
    "    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(int(bar.get_height())), \n",
    "            ha='center', va='bottom', fontsize=12, fontweight='bold', color='dimgrey')\n",
    "\n",
    "ax.grid(True, which='major', linestyle='--', linewidth=0.7)\n",
    "\n",
    "plt.tight_layout()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "05b82ad1-2f7d-40bb-90c7-cee9ec0db91d",
   "metadata": {},
   "source": [
    "-----\n",
    "## Age Distribution Heatmap\n",
    "- **Outcome Comparison**: Died vs. Survived."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d71bdfd-3cd7-4528-8675-25264ed37d5b",
   "metadata": {
    "panel-layout": {
     "height": 1162,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Filter passengers who died and survived\n",
    "died_df = df[df['Survived'] == 0]\n",
    "survived_df = df[df['Survived'] == 1]\n",
    "\n",
    "# Define age bins\n",
    "age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]\n",
    "\n",
    "# Group ages into bins and count occurrences\n",
    "died_age_bins = pd.cut(died_df['Age'], bins=age_bins).value_counts().sort_index().fillna(0)\n",
    "survived_age_bins = pd.cut(survived_df['Age'], bins=age_bins).value_counts().sort_index().fillna(0)\n",
    "\n",
    "# Create a DataFrame for the heatmap\n",
    "heatmap_data = pd.DataFrame({'Died': died_age_bins, 'Survived': survived_age_bins})\n",
    "\n",
    "# Create the heatmap\n",
    "plt.figure(figsize=(12, 8))  # Adjust th e figure size\n",
    "sns.heatmap(heatmap_data.T, cmap=\"coolwarm\", linewidths=0.5, linecolor='lightgray', annot=True, fmt='g', cbar=False)  # Use seaborn for heatmap\n",
    "\n",
    "# Set plot title and labels with bold font weight\n",
    "plt.title('Age Distribution of Passengers Who Died and Survived', fontsize=20, fontweight='bold', pad=20)  # Increase title font size and set bold\n",
    "plt.xlabel('Age Group', fontsize=14, fontweight='bold', labelpad=15)  # Increase x-axis label font size and set bold\n",
    "plt.ylabel('Outcome', fontsize=14, fontweight='bold', labelpad=15)  # Increase y-axis label font size and set bold\n",
    "\n",
    "plt.xticks(fontsize=12, fontweight='bold')  # Increase x-axis tick label font size and set bold\n",
    "plt.yticks(fontsize=12, fontweight='bold', rotation=0)  # Increase y-axis tick label font size and set bold\n",
    "\n",
    "plt.tight_layout()  # Adjust layout to prevent overlapping elements\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "60dc12e2-b53e-4370-bb3c-ff6db915a66a",
   "metadata": {},
   "source": [
    "----\n",
    "## Death Count by Passenger Class \n",
    "- **Passenger Class**: 1st, 2nd, and 3rd."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f259ff75",
   "metadata": {
    "panel-layout": {
     "height": 874,
     "visible": true,
     "width": 100
    },
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Filter passengers who did not survive\n",
    "not_survived_df = df[df['Survived'] == 0]\n",
    "\n",
    "# Count deaths in each passenger class\n",
    "death_count_by_pclass = not_survived_df['Pclass'].value_counts().sort_index()\n",
    "\n",
    "# Define colors for each bar\n",
    "colors = ['gray', 'gray', 'gray']\n",
    "\n",
    "# Create a bar plot\n",
    "plt.figure(figsize=(8, 6))  # Adjust the figure size\n",
    "bars = death_count_by_pclass.plot(kind='bar', color=colors ,edgecolor='black')  # Plot the bar plot\n",
    "\n",
    "# Annotate each bar with its count\n",
    "for bar in bars.patches:\n",
    "    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(int(bar.get_height())), \n",
    "             ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')\n",
    "\n",
    "# Set plot title and labels with bold font weight\n",
    "plt.title('Death Count by Passenger Class', fontsize=16, fontweight='bold')  # Increase title font size and set bold\n",
    "plt.xlabel('Passenger Class', fontsize=14, fontweight='bold')  # Increase x-axis label font size and set bold\n",
    "plt.ylabel('Death Count', fontsize=14, fontweight='bold')  # Increase y-axis label font size and set bold\n",
    "\n",
    "# Customize ticks and tick labels with bold font weight\n",
    "plt.xticks(rotation=0, fontsize=12, fontweight='bold')  # Rotate x-axis tick labels to horizontal and set font size and bold\n",
    "plt.yticks(fontsize=12, fontweight='bold')  # Set y-axis tick label font size and bold\n",
    "\n",
    "plt.tight_layout()  # Adjust layout to prevent overlapping elements\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e17bf96-3a46-44ea-bf8f-2224b64a4b25",
   "metadata": {},
   "source": [
    "# -----Feature Engineering-----"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4fe2ee34",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a binary column indicating cabin presence for the training and testing dataset\n",
    "df[\"CabinBool\"] = (df[\"Cabin\"].notnull().astype('int')) \n",
    "dft[\"CabinBool\"] = (dft[\"Cabin\"].notnull().astype('int')) \n",
    "\n",
    "#drop Cabin because we dont need it anymore  \n",
    "df = df.drop(['Cabin'], axis=1) \n",
    "dft = dft.drop(['Cabin'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ad58357",
   "metadata": {
    "panel-layout": {
     "height": 1307.2000732421875,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "df.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6b7113c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Remove the 'Ticket' column from the training and testing dataset because its useless\n",
    "df = df.drop(['Ticket'], axis=1) \n",
    "dft = dft.drop(['Ticket'], axis=1) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b72dff06",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Fill missing values in the 'Age' column with -1\n",
    "df[\"Age\"] = df[\"Age\"].fillna(-1) \n",
    "dft[\"Age\"] = dft[\"Age\"].fillna(-1) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aba0b4e6-3e36-4084-b8ca-ab8e2332037c",
   "metadata": {},
   "source": [
    "### Age Group Binning\n",
    "#### - **Unknown**: (-1, 0)\n",
    "#### - **Baby**: (0, 5)\n",
    "#### - **Child**: (5, 12)\n",
    "#### - **Teenager**: (12, 18)\n",
    "#### - **Student**: (18, 24)\n",
    "#### - **Young Adult**: (24, 35)\n",
    "#### - **Adult**: (35, 60)\n",
    "#### - **Senior**: (60, Infinite)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9686ce62",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define age bins and labels\n",
    "bins=[-1,0,5,12,18,24,35,60,np.inf]\n",
    "labels=['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6bd5e8d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create age groups based on defined bins and labels for the training and testing dataset\n",
    "df['AgeGroup'] = pd.cut(df['Age'], bins, labels=labels)\n",
    "dft['AgeGroup'] = pd.cut(dft['Age'], bins, labels=labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bba48e67-007f-42f7-b352-4c9e79c46b72",
   "metadata": {},
   "source": [
    "----\n",
    "### Title Transformation \n",
    "### - Extract titles from names."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "268b292a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#Combine the train and test datasets\n",
    "combine = [df, dft]\n",
    "#Extract titles from names and categorize them\n",
    "for dataset in combine:\n",
    "    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)\n",
    "    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')\n",
    "    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')\n",
    "    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')\n",
    "    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')\n",
    "    \n",
    "#Display the cross-tabulation of titles and sex \n",
    "print(pd.crosstab(df['Title'], df['Sex']))\n",
    "#Calculate the survival rate for each title\n",
    "print(df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())\n",
    "#Map titles to numerical values\n",
    "title_mapping = {\"Mr\": 1, \"Miss\": 2, \"Mrs\": 3, \"Master\": 4, \"Royal\": 5, \"Rare\": 6}\n",
    "for dataset in combine:\n",
    "    dataset['Title'] = dataset['Title'].map(title_mapping).fillna(0)\n",
    "df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8822125c-f265-4e11-bd84-48d7bb07d0e5",
   "metadata": {},
   "source": [
    "----\n",
    "## Age Group Imputation from Titles"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1675fc80",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Mapping numerical age group to labels\n",
    "age_title_mapping = {\n",
    "    1: \"Baby\",\n",
    "    2: \"Child\",\n",
    "    3: \"Teenager\",\n",
    "    4: \"Student\",\n",
    "    5: \"Young Adult\",\n",
    "    6: \"Adult\",\n",
    "    7: \"Senior\"\n",
    "}\n",
    "#Fill missing age group values based on title\n",
    "#If AgeGroup is missing, map it to a descriptive label based on Title\n",
    "df.loc[df[\"AgeGroup\"].isna(), \"AgeGroup\"] = df.loc[df[\"AgeGroup\"].isna(), \"Title\"].map(age_title_mapping)\n",
    "\n",
    "dft.loc[dft[\"AgeGroup\"].isna(), \"AgeGroup\"] = dft.loc[dft[\"AgeGroup\"].isna(), \"Title\"].map(age_title_mapping)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0c0481e5-4e9c-485c-8f51-c2e19a8d06e1",
   "metadata": {},
   "source": [
    "----\n",
    "## Age Group Mapping"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9aef3991",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define a dictionary to map age group labels to numerical values\n",
    "age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, \n",
    "'Student': 4, 'Young Adult': 5, 'Adult': 6, \n",
    "'Senior': 7} \n",
    "#Map age group labels to numerical values for the training and testing dataset\n",
    "df['AgeGroup'] = df['AgeGroup'].map(age_mapping) \n",
    "dft['AgeGroup'] = dft['AgeGroup'].map(age_mapping) \n",
    " \n",
    "# Drop the 'Age' column from the training and testing dataset\n",
    "df = df.drop(['Age'], axis=1) \n",
    "dft = dft.drop(['Age'], axis=1) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e2de2f7b",
   "metadata": {
    "panel-layout": {
     "height": 1307.2000732421875,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "df.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54a33b50",
   "metadata": {},
   "outputs": [],
   "source": [
    "#drop name because its useless now\n",
    "df = df.drop(['Name'], axis=1) \n",
    "dft = dft.drop(['Name'], axis=1) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "22df47c0-51a2-4d45-9afc-2e9e70539fa1",
   "metadata": {},
   "source": [
    "----------\n",
    "## Gender Mapping"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55a9db58",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Map 'female' to 1 and 'male' to 0.\n",
    "sex_mapping ={\"female\": 1, \"male\": 0}\n",
    "#Apply the mapping to the 'Sex' column in both the training and testing dataset\n",
    "df['Sex'] = df['Sex'].map(sex_mapping) \n",
    "dft['Sex'] = dft['Sex'].map(sex_mapping) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b9bf21bf-b7bd-42a5-ab0c-f9d6129ad34d",
   "metadata": {},
   "source": [
    "------\n",
    "### Embarked Mapping\n",
    "\n",
    "- Map 'S' to 1, 'C' to 2, and 'Q' to 3."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11b02a98",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Mapping of Embarked port codes to numerical values\n",
    "embarked_mapping = {\"S\": 1, \"C\": 2, \"Q\": 3}\n",
    "#Apply the mapping to the 'Embarked' column in both the training and test datasets\n",
    "df['Embarked'] = df['Embarked'].map(embarked_mapping) \n",
    "dft['Embarked'] = dft['Embarked'].map(embarked_mapping)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ab3d399",
   "metadata": {
    "panel-layout": {
     "height": 1307.2000732421875,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "19bfe3ca-105b-4d2f-abf8-d85e15f6c2c8",
   "metadata": {},
   "source": [
    "-----\n",
    "## Fare Data Processing\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e28fb81b",
   "metadata": {
    "panel-layout": {
     "height": 874,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "#Fill missing 'Fare' values in the test set using the average fare of each passenger class\n",
    "dft['Fare'] = dft['Fare'].fillna(dft.groupby('Pclass')['Fare'].transform('mean').round(4))\n",
    "# Create 'FareBand' by dividing 'Fare' into 4 equal parts (quartiles)\n",
    "# Assign labels 1, 2, 3, 4 to these parts\n",
    "df['FareBand'] = pd.qcut(df['Fare'], 4, labels=[1, 2, 3, 4])\n",
    "dft['FareBand'] = pd.qcut(dft['Fare'], 4, labels=[1, 2, 3, 4])\n",
    "#Remove the original 'Fare' column since we now have 'FareBand'\n",
    "df.drop('Fare', axis=1, inplace=True)\n",
    "dft.drop('Fare', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ca1a5d8",
   "metadata": {
    "panel-layout": {
     "height": 1043.2000732421875,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "dft.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ef7f7030-022b-4db0-a765-e14b23c97486",
   "metadata": {},
   "source": [
    "# -----Model Training-----\n",
    "## Random Forest Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f1c888f",
   "metadata": {
    "panel-layout": {
     "height": 874,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix\n",
    "\n",
    "#Define the feature columns for the model\n",
    "#These are the columns that will be used to predict survival\n",
    "X_rf = df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'CabinBool', 'AgeGroup', 'Title', 'FareBand']]\n",
    "y_rf = df['Survived']\n",
    "\n",
    "#Define the target column\n",
    "#This is the column that the model will try to predict (whether the passenger survived or not)\n",
    "X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)\n",
    "\n",
    "#Split the dataset into training and validation sets\n",
    "#80% of the data will be used for training and 20% will be used for validation\n",
    "model_rf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "\n",
    "#Train the model on the training data\n",
    "model_rf.fit(X_train_rf, y_train_rf)\n",
    "\n",
    "#Predict the survival on the validation set\n",
    "y_pred_rf = model_rf.predict(X_val_rf)\n",
    "\n",
    "#Define the test set features (from the test dataset)\n",
    "X_test_rf = dft[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'CabinBool', 'AgeGroup', 'Title', 'FareBand']]\n",
    "\n",
    "#Predict the survival on the test set\n",
    "y_pred_test_rf = model_rf.predict(X_test_rf)\n",
    "\n",
    "#Calculate the accuracy of the model on the validation set\n",
    "accuracy_rf = accuracy_score(y_val_rf, y_pred_rf)\n",
    "\n",
    "print(\"Random Forest Validation Accuracy:\", accuracy_rf)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a07ee1ce-77a9-42a8-94e8-0f8acd672c28",
   "metadata": {},
   "source": [
    "------\n",
    "## Support Vector Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10f286e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC\n",
    "#Define the feature columns for the SVC model\n",
    "#These are the columns that will be used to predict survival\n",
    "X_svc = df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'CabinBool', 'AgeGroup', 'Title', 'FareBand']]\n",
    "\n",
    "#Define the target column\n",
    "#This is the column that the model will try to predict (whether the passenger survived or not)\n",
    "y_svc = df['Survived']\n",
    "\n",
    "#Split the dataset into training and validation sets\n",
    "#80% of the data will be used for training and 20% will be used for validation\n",
    "X_train_svc, X_val_svc, y_train_svc, y_val_svc = train_test_split(X_svc, y_svc, test_size=0.2, random_state=42)\n",
    "\n",
    "#Initialize the Support Vector Classifier (SVC)\n",
    "#We use a radial basis function (rbf) kernel and set a random state for reproducibility\n",
    "model_svc = SVC(kernel='rbf', random_state=42)\n",
    "\n",
    "#Train the model on the training data\n",
    "model_svc.fit(X_train_svc, y_train_svc)\n",
    "\n",
    "#Predict the survival on the validation set\n",
    "y_pred_svc = model_svc.predict(X_val_svc)\n",
    "\n",
    "#Calculate the accuracy of the model on the validation set\n",
    "accuracy_svc = accuracy_score(y_val_svc, y_pred_svc)\n",
    "\n",
    "#Print the validation accuracy\n",
    "print(\"SVC Validation Accuracy:\", accuracy_svc)\n",
    "\n",
    "#Define the test set features (from the test dataset)\n",
    "X_test_svc = dft[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'CabinBool', 'AgeGroup', 'Title', 'FareBand']]\n",
    "\n",
    "#Predict the survival on the test set\n",
    "y_pred_test_svc = model_svc.predict(X_test_svc)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0534415b-aeae-4a3d-9296-10a5ee25607e",
   "metadata": {},
   "source": [
    "-----\n",
    "## K-Nearest Neighbors Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "71ff92dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "#Define the feature columns for the KNN model\n",
    "#These are the columns that will be used to predict survival\n",
    "X_knn = df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'CabinBool', 'AgeGroup', 'Title', 'FareBand']]\n",
    "\n",
    "#Define the target column\n",
    "#This is the column that the model will try to predict (whether the passenger survived or not)\n",
    "y_knn = df['Survived']\n",
    "\n",
    "#Split the dataset into training and validation sets\n",
    "#80% of the data will be used for training and 20% will be used for validation\n",
    "X_train_knn, X_val_knn, y_train_knn, y_val_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)\n",
    "\n",
    "#Initialize the StandardScaler\n",
    "#This will standardize the features by removing the mean and scaling to unit variance\n",
    "scaler_knn = StandardScaler()\n",
    "\n",
    "#Fit the scaler on the training data and transform the training data\n",
    "X_train_scaled_knn = scaler_knn.fit_transform(X_train_knn)\n",
    "\n",
    "#Transform the validation data using the same scaler\n",
    "X_val_scaled_knn = scaler_knn.transform(X_val_knn)\n",
    "\n",
    "#Initialize the K-Nearest Neighbors Classifier\n",
    "#The default number of neighbors is 5\n",
    "model_knn = KNeighborsClassifier()\n",
    "\n",
    "#Train the model on the scaled training data\n",
    "model_knn.fit(X_train_scaled_knn, y_train_knn)\n",
    "\n",
    "#Predict the survival on the scaled validation set\n",
    "y_pred_knn = model_knn.predict(X_val_scaled_knn)\n",
    "\n",
    "#Calculate the accuracy of the model on the validation set\n",
    "accuracy_knn = accuracy_score(y_val_knn, y_pred_knn)\n",
    "\n",
    "#Print the validation accuracy\n",
    "print(\"KNN Validation Accuracy:\", accuracy_knn)\n",
    "\n",
    "#Define the test set features (from the test dataset)\n",
    "X_test_knn = dft[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'CabinBool', 'AgeGroup', 'Title', 'FareBand']]\n",
    "\n",
    "#Transform the test set features using the same scaler\n",
    "X_test_scaled_knn = scaler_knn.transform(X_test_knn)\n",
    "\n",
    "#Predict the survival on the scaled test set\n",
    "y_pred_test_knn = model_knn.predict(X_test_scaled_knn)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd47290e-5d36-4e5f-9822-f3cb79df9ba5",
   "metadata": {},
   "source": [
    "-------\n",
    "## Random Forest Classifier Evaluation\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a5fc26ca",
   "metadata": {
    "panel-layout": {
     "height": 874,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "accuracy_rf = accuracy_score(y_val_rf, y_pred_rf)\n",
    "precision_rf = precision_score(y_val_rf, y_pred_rf)\n",
    "recall_rf = recall_score(y_val_rf, y_pred_rf)\n",
    "f1_rf = f1_score(y_val_rf, y_pred_rf)\n",
    "conf_matrix_rf = confusion_matrix(y_val_rf, y_pred_rf)\n",
    "\n",
    "print(\"Random Forest Validation Accuracy:\", accuracy_rf)\n",
    "print(\"Random Forest Precision:\", precision_rf)\n",
    "print(\"Random Forest Recall:\", recall_rf)\n",
    "print(\"Random Forest F1 Score:\", f1_rf)\n",
    "print(\"Random Forest Confusion Matrix:\")\n",
    "print(conf_matrix_rf)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "57902de0-77b4-4d88-a80b-659e9566e29d",
   "metadata": {},
   "source": [
    "------\n",
    "## Support Vector Classifier Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "103f28ab",
   "metadata": {
    "panel-layout": {
     "height": 874,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "accuracy_svc = accuracy_score(y_val_svc, y_pred_svc)\n",
    "precision_svc = precision_score(y_val_svc, y_pred_svc)\n",
    "recall_svc = recall_score(y_val_svc, y_pred_svc)\n",
    "f1_svc = f1_score(y_val_svc, y_pred_svc)\n",
    "conf_matrix_svc = confusion_matrix(y_val_svc, y_pred_svc)\n",
    "\n",
    "print(\"SVC Validation Accuracy:\", accuracy_svc)\n",
    "print(\"SVC Precision:\", precision_svc)\n",
    "print(\"SVC Recall:\", recall_svc)\n",
    "print(\"SVC F1 Score:\", f1_svc)\n",
    "print(\"SVC Confusion Matrix:\")\n",
    "print(conf_matrix_svc)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0772dd0d-83e5-4b5f-a32b-59e46cb5fb4e",
   "metadata": {},
   "source": [
    "-----\n",
    "## K-Nearest Neighbors Classifier Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f81dc88d",
   "metadata": {
    "panel-layout": {
     "height": 874,
     "visible": true,
     "width": 100
    }
   },
   "outputs": [],
   "source": [
    "accuracy_knn = accuracy_score(y_val_knn, y_pred_knn)\n",
    "precision_knn = precision_score(y_val_knn, y_pred_knn)\n",
    "recall_knn = recall_score(y_val_knn, y_pred_knn)\n",
    "f1_knn = f1_score(y_val_knn, y_pred_knn)\n",
    "conf_matrix_knn = confusion_matrix(y_val_knn, y_pred_knn)\n",
    "\n",
    "print(\"KNN Validation Accuracy:\", accuracy_knn)\n",
    "print(\"KNN Precision:\", precision_knn)\n",
    "print(\"KNN Recall:\", recall_knn)\n",
    "print(\"KNN F1 Score:\", f1_knn)\n",
    "print(\"KNN Confusion Matrix:\")\n",
    "print(conf_matrix_knn)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  },
  "panel-cell-order": [
   "8b607bc8",
   "6b2e840c",
   "fb4b29fa",
   "e114c0de",
   "8e227307",
   "5fe0df74",
   "98f3f85b",
   "ab648359",
   "6d8c3791",
   "cd6a6465",
   "203f9537",
   "bffc668f",
   "eff1b09c",
   "92befd5a",
   "1fd8a8e0",
   "0d3b6d01",
   "b78763e4",
   "f259ff75",
   "5ad58357",
   "e2de2f7b",
   "2ab3d399",
   "e28fb81b",
   "9ca1a5d8",
   "0f1c888f",
   "a5fc26ca",
   "103f28ab",
   "f81dc88d"
  ]
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
