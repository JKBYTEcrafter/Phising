{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {}, 
   "source": [
    "### * **What is a phishing attack?**\n",
    "* Phishing is a type of social engineering attack often used to steal user data, including login credentials and credit card numbers. It occurs when an attacker, masquerading as a trusted entity, dupes a victim into opening an email, instant message, or text message. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### * Phishing attack examples\n",
    "* A spoofed email ostensibly from myuniversity.edu is mass-distributed to as many faculty members as possible. The email claims that the user’s password is about to expire. Instructions are given to go to myuniversity.edu/renewal to renew their password within 24 hours.>\n",
    "<img src='https://github.com/taruntiwarihp/raw_images/blob/master/phishing-attack-email-example.png?raw=True'>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Several things can occur by clicking the link. For example:\n",
    "\n",
    "    1. The user is redirected to myuniversity.edurenewal.com, a bogus page appearing exactly like the real renewal page, where both new and existing passwords are requested. The attacker, monitoring the page, hijacks the original password to gain access to secured areas on the university network.\n",
    "    \n",
    "    2. The user is sent to the actual password renewal page. However, while being redirected, a malicious script activates in the background to hijack the user’s session cookie. This results in a reflected XSS attack, giving the perpetrator privileged access to the university network."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### * Importing some useful libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": True
   },
   "outputs": [],
   "source": [
    "import pandas as pd # use for data manipulation and analysis\n",
    "import numpy as np # use for multi-dimensional array and matrix\n",
    "\n",
    "import seaborn as sns # use for high-level interface for drawing attractive and informative statistical graphics \n",
    "import matplotlib.pyplot as plt # It provides an object-oriented API for embedding plots into applications\n",
    "%matplotlib inline \n",
    "# It sets the backend of matplotlib to the 'inline' backend:\n",
    "import time # calculate time \n",
    "\n",
    "from sklearn.linear_model import LogisticRegression # algo use to predict good or bad\n",
    "from sklearn.naive_bayes import MultinomialNB # nlp algo use to predict good or bad\n",
    "\n",
    "from sklearn.model_selection import train_test_split # spliting the data between feature and target\n",
    "from sklearn.metrics import classification_report # gives whole report about metrics (e.g, recall,precision,f1_score,c_m)\n",
    "from sklearn.metrics import confusion_matrix # gives info about actual and predict\n",
    "from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text  \n",
    "from nltk.stem.snowball import SnowballStemmer # stemmes words\n",
    "from sklearn.feature_extraction.text import CountVectorizer # create sparse matrix of words using regexptokenizes  \n",
    "from sklearn.pipeline import make_pipeline # use for combining all prerocessors techniuqes and algos\n",
    "\n",
    "from PIL import Image # getting images in notebook\n",
    "# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator# creates words colud\n",
    "\n",
    "from bs4 import BeautifulSoup # use for scraping the data from website\n",
    "from selenium import webdriver # use for automation chrome \n",
    "import networkx as nx # for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.\n",
    "\n",
    "import pickle# use to dump model \n",
    "\n",
    "import warnings # ignores pink warnings \n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* **Did some surfing and found some websites offering malicious links.** And found some datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": True
   },
   "outputs": [],
   "source": [
    "phishing_data1 = pd.read_csv('phishing_urls.csv',usecols=['domain','label'],encoding='latin1', error_bad_lines=False)\n",
    "phishing_data1.columns = ['URL','Label']\n",
    "phishing_data2 = pd.read_csv('phishing_data.csv')\n",
    "phishing_data2.columns = ['URL','Label']\n",
    "phishing_data3 = pd.read_csv('phishing_data2.csv')\n",
    "phishing_data3.columns = ['URL','Label']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "for l in range(len(phishing_data1.Label)):\n",
    "    if phishing_data1.Label.loc[l] == '1.0':\n",
    "        phishing_data1.Label.loc[l] = 'bad'\n",
    "    else:\n",
    "        phishing_data1.Label.loc[l] = 'good'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* **Concatenate All datasets in one.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "frames = [phishing_data1, phishing_data2, phishing_data3]\n",
    "phishing_urls = pd.concat(frames)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "#saving dataset\n",
    "phishing_urls.to_csv(r'phishing_site_urls.csv', index = False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* **Loading the main dataset.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "phish_data = pd.read_csv('phishing_site_urls.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### * You can download dataset from my **Kaggle** Profile <a href='https://www.kaggle.com/taruntiwarihp/phishing-site-urls'>here</a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": False
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>URL</th>\n",
       "      <th>Label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>nobell.it/70ffb52d079109dca5664cce6f317373782/...</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>www.dghjdgf.com/paypal.co.uk/cycgi-bin/webscrc...</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>serviciosbys.com/paypal.cgi.bin.get-into.herf....</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>mail.printakid.com/www.online.americanexpress....</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>thewhiskeydregs.com/wp-content/themes/widescre...</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                 URL Label\n",
       "0  nobell.it/70ffb52d079109dca5664cce6f317373782/...   bad\n",
       "1  www.dghjdgf.com/paypal.co.uk/cycgi-bin/webscrc...   bad\n",
       "2  serviciosbys.com/paypal.cgi.bin.get-into.herf....   bad\n",
       "3  mail.printakid.com/www.online.americanexpress....   bad\n",
       "4  thewhiskeydregs.com/wp-content/themes/widescre...   bad"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "phish_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>URL</th>\n",
       "      <th>Label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>549341</td>\n",
       "      <td>23.227.196.215/</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>549342</td>\n",
       "      <td>apple-checker.org/</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>549343</td>\n",
       "      <td>apple-iclods.org/</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>549344</td>\n",
       "      <td>apple-uptoday.org/</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>549345</td>\n",
       "      <td>apple-search.info</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       URL Label\n",
       "549341     23.227.196.215/   bad\n",
       "549342  apple-checker.org/   bad\n",
       "549343   apple-iclods.org/   bad\n",
       "549344  apple-uptoday.org/   bad\n",
       "549345   apple-search.info   bad"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "phish_data.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": False
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 549346 entries, 0 to 549345\n",
      "Data columns (total 2 columns):\n",
      "URL      549346 non-None object\n",
      "Label    549346 non-None object\n",
      "dtypes: object(2)\n",
      "memory usage: 8.4+ MB\n"
     ]
    }
   ],
   "source": [
    "phish_data.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* **About dataset**\n",
    "* Data is containg 5,49,346 unique entries.\n",
    "* There are two columns.\n",
    "* Label column is prediction col which has 2 categories \n",
    "    A. Good - which means the urls is not containing malicious stuff and **this site is not a Phishing Site.**\n",
    "    B. Bad - which means the urls contains malicious stuffs and **this site isa Phishing Site.**\n",
    "* There is no missing value in the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "scrolled": True
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "URL      0\n",
       "Label    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "phish_data.isNone().sum() # there is no missing values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* **Since it is classification problems so let's see the classes are balanced or imbalances**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "#create a dataframe of classes counts\n",
    "label_counts = pd.DataFrame(phish_data.Label.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x15e4f1821f0>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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cePG8fnnn/Pkk09SUlLCww8/zDfffMP8+fPZtGkTW7Zs4Xe/+11zjFVERKROjX5rfkREBK+99hrvvPMOrVu3ZsCAAUydOtVYFdipUyfmz5/f5AMVERG5nkYvALme77//nqKiIgoKCsjKymL8+PFN0e0dSwtApDloAYjcqW7ZG0A6depEr1692L9/P0uXLm2qbkVEROrVLK+zEhERuZUUZiIiYnoKMxERMT2FmYiImF69S/O///77RnVYWlp604MRERG5GfWG2eDBgxv1Rg+73a43gIiIyC1Vb5iNGjVK4SQiIre1esNs2bJlt2IcIiIiN00LQERExPQUZiIiYnqNftGwiEh9vNq2xOLm7uxhyG2mqvwnLhVVNEvfCjMRaXIWN3dy/1t/Ckoc+bySATRPmOk2o4iImJ7CTERETE9hJiIipqcwExER01OYiYiI6SnMRETE9BRmIiJiegozERExPYWZiIiYnsJMRERMT2EmIiKmpzATERHTU5iJiIjpKcxERMT0FGYiImJ6CjMRETE9hZmIiJiewkxERExPYSYiIqanMBMREdNTmImIiOkpzERExPRumzDLzMzkt7/9LT/88INDe2pqKmPGjMHf35/BgwezefPmWp/NyMggKiqKgIAABg4cyIoVK6ioqHCoOXv2LNOnTycoKIj+/fvz6quvUlJS4lBz8eJFYmJi6N+/P4GBgcyePZsLFy441JSWlrJw4UJCQkIICAhg6tSpnD17tmlOgoiI3JTbIsyysrKYNm0alZWVDu3p6elMnz6drl278uabb/LII48QHx/P22+/bdTk5OQwceJEWrVqxapVq5g0aRLvvPMOS5cuNWqKioqYMGECFy9e5PXXXycmJoY9e/YQExNj1FRWVjJ58mS+/PJL4uLiiIuLIz09nSlTpjiMa9asWXz22WfMmTOH119/nfPnzzN+/HguX77cjGdIRERupIUzD15ZWcmf//xnli9fTsuWLWvtT0xMpHfv3iQkJAAQGhpKZWUl69evJyoqCjc3NzZu3IiHhwfr1q3Dzc2NsLAw3N3dWbRoEdOmTcPb25vt27dTXFzM7t278fT0BMDb25vo6GhOnDiBv78/SUlJfPvtt+zZs4du3boB0KtXL0aMGEFycjKRkZGkpaWRkpLCpk2bCA0NBSAoKIjw8HB27NhBdHT0LTpzIiLyc06dmR07dow33niDSZMmMWfOHId9V69eJS0tjaFDhzq0Dxs2jOLiYtLT0wE4fPgwgwYNws3NzaiJiIigqqqK1NRUoyY4ONgIMoCBAwditVpJSUkxarp3724EGWBs/7zGarUSEhJi1Hh5eREcHMyhQ4ea4pSIiMhNcGqYdevWjQMHDvDcc89hsVgc9p07d46Kigr8/Pwc2n19fQHIzs7mypUr5Ofn16rx8vLCZrORnZ0NXLuN+a81FouFLl263LAGwMfHx6HG19e31lh/XiMiIreeU8Psrrvuon379nXuq3kGZbPZHNqtVisAJSUl162pqatZ4HH58uUmqSkpKam3RkREbj2nPjO7EbvdDoCLi0ud+11dXW9YY7fbcXX9Z1Y3RU3N8a43nsZo3752KIo0hQ4dPJw9BJHraq7r87YNMw+Pa1/4X2c8NdseHh7GLKmuWVFZWZnRh81mq7OmtLSUzp0711tTcxybzUZeXt4NaxqqoKCE6urrh2N99B+WXM+FC85fWavrU67nZq9PV1eXG04Cboul+XXx8fHBYrGQm5vr0F6z7efnh9Vqxdvbm5ycHIeagoICSkpKjGdgfn5+tWqqqqrIy8u7YU3N8X5ec+7cuVoztJycnDqft4mIyK1x24ZZq1atCAoKIjk52SE89u3bh4eHB3369AEgJCSEgwcPUl5e7lBjsVjo16+fUXP06FEKCwuNmtTUVMrKyhgwYABwbXXjqVOnyMrKMmpOnz5NVlaWQ01xcTFHjhwxai5dukRaWppRIyIit95tG2YAzzzzDOnp6cyaNYuUlBRWrVrF22+/zbRp02jdujUAU6ZM4cKFC0RHR3Pw4EHjB9P/+Z//SadOnQB48skncXNzY+LEiezfv58PPviAF198kdDQUPr27QtAZGQkvr6+TJkyhaSkJD799FOmTp3Kv//7vzN8+HAAgoOD6devH7Nnz+aDDz5g//79TJw4EQ8PD5544gnnnCQREcHFfqNVDbfQ//zP/zB//nxSUlLo2LGj0b5//34SExPJzs7G29ubcePGMWnSJIfPpqWlER8fT2ZmJp6enowaNYoZM2Y4/BD75MmTLFmyhOPHj2O1WhkyZAhz5851eNaVn5/P4sWLOXz4MG5uboSEhBAbG8vdd99t1BQVFbFs2TIOHDhAdXU1gYGBxMbG0rVr10Z936Z4Zhb44tab/rzcmY4ljL9tnpnl/vfvnD0Muc34vJLRbM/Mbpsw+7VRmElzUJjJ7aw5w+y2vs0oIiLSEAozERExPYWZiIiYnsJMRERMT2EmIiKmpzATERHTU5iJiIjpKcxERMT0FGYiImJ6CjMRETE9hZmIiJiewkxERExPYSYiIqanMBMREdNTmImIiOkpzERExPQUZiIiYnoKMxERMT2FmYiImJ7CTERETE9hJiIipqcwExER01OYiYiI6SnMRETE9BRmIiJiegozERExPYWZiIiYnsJMRERMT2EmIiKmpzATERHTU5iJiIjpKcxERMT0FGYiImJ6CjMRETE9hZmIiHQgbmgAAAeoSURBVJiewkxERExPYSYiIqanMBMREdNTmImIiOkpzERExPQUZjfh008/5eGHH+a+++5j+PDh7N6929lDEhH5VVOYNdLevXuZM2cOISEhrF27ln79+jFv3jw+++wzZw9NRORXq4WzB2A2K1asYPjw4bz00ksAPPjggxQVFbF69WoiIiKcPDoRkV8nzcwa4dy5c+Tm5jJ06FCH9mHDhpGVlcW5c+ecNDIRkV83zcwaISsrCwA/Pz+Hdl9fXwCys7O55557GtSXq6vLLx7Pbzytv7gPufM0xbXVFCxtOzl7CHIbutnrs77PKcwa4fLlywDYbDaHdqv1WqiUlJQ0uC/PJgiiT18a84v7kDtP+/a2+otugc4z9zl7CHIbaq7rU7cZG8FutwPg4uJSZ7urq06niIgz6H/fRvDw8ABqz8BKS0sd9ouIyK2lMGuEmmdlubm5Du05OTkO+0VE5NZSmDWCr68vXbp0qfWbsuTkZO699146ddIDbxERZ9ACkEZ69tlnmT9/Pm3btuWhhx7if//3f9m7dy8rV6509tBERH61XOw1qxekwd5//302b95Mfn4+99xzD9HR0YwaNcrZwxIR+dVSmImIiOnpmZmIiJiewkxERExPYSa/aj179mTdunXOHoaYxODBg/njH//YLH3rWvxlFGYiImJ6CjMRETE9hZk4VXl5OcuWLWPgwIHcf//9zJw5ky1bttCzZ0+jZvfu3YwePZr777+f0NBQXn/9dX766SeHflJSUviv//ovAgICeOCBB1iwYAE//vijQ83f//53xo4di7+/P8OGDePIkSO35DvKnaW8vJxXX32VwMBA+vfvT1xcnPGKu6qqKjZs2MCIESO47777uP/++3niiSc4evSoQx+6Fpuewkyc6uWXX2bHjh1MnjyZ1atXU15ezvLly439iYmJxMbGEhwczJo1a3j66ad5//33mT59uvGC5507dxIdHY2Pjw+rV69m1qxZHDx4kPHjx3PlyhUAvv76ayZNmoSHhweJiYmMHz+e2bNnO+U7i7klJSWRlZXFG2+8wXPPPcdHH31k/LHe+Ph41q9fzxNPPMFbb73Fa6+9xo8//sjMmTN1LTY3u4iT5OTk2Hv27Gnftm2b0VZVVWV/+OGH7T169LD/+OOP9j59+tgXLlzo8LmkpCR7jx497AcPHrRXVVXZBwwYYI+OjnaoOXHihL1Hjx5G3zNmzLA/9NBD9vLy8lr9rF27thm/pdxJBg0aZA8JCbFfuXLFaHvvvffsPXv2tJ86dcr+wgsv2Ldu3erwmX379tl79OhhP3HihN1u17XYXDQzE6c5evQodrvd4S93u7q6EhERAcCJEycoLy/n4YcfdvhcREQELVu25OjRo2RnZ3Px4sVaNffddx++vr7G7Z1jx47x4IMP0rJlS6Nm6NChWCyW5vp6cocKCwvD3d3d2A4PD8dut5ORkcHKlSuJiori0qVLpKWlsXPnTj7++GMAKioqAF2LzUXvZhSnuXTpEgBeXl4O7XfddRcARUVFAHTo0MFhv6urK15eXpSUlFBYWFhnDUD79u2NZxlFRUW1jtOiRQs8PT2b4JvIr0nN9Vmj5rr6xz/+QUZGBgsXLiQjI4PWrVvTvXt34wXk9v9/W1zXYvPQzEycxtvbG4CCggKH9prttm3bAnDhwgWH/dXV1Vy6dAlPT8/r1tS01fwH0a5du1rHsdvtRmCKNNS/XjM115W7uztTpkzBZrORlJREeno6H374IWPGOP5FeF2LzUNhJk7Tt29fLBYLn3/+uUN7zba/vz9ubm4kJSU57P/ss8+oqKggMDCQrl27ctddd9Wq+fLLLzl37hx9+/YF4IEHHuDgwYMOqyD/9re/Gbd+RBrqiy++oKqqytjeu3cvAAEBARQWFjJx4kS6d+9u/OX5Q4cOAf+cmelabB66zShO4+Pjw6OPPkp8fDxXr16lW7du7Nq1i8zMTFxcXGjXrh2TJ09m/fr1tGjRgrCwME6dOsWbb75Jv379ePDBB3F1deWFF15gwYIFzJ07lxEjRnD+/HlWr16Nn58fo0ePBq796Z4DBw4wdepUJk2axMWLF1m9erXDcwuRhvjhhx+YNWsWTzzxBJmZmaxatYrHHnsMPz8/bDYb69atw8XFBVdXV5KTk/nwww8BKCsrA3QtNhe9NV+c6qeffiIhIYFPP/2Uq1evEh4eTps2bfjoo49IT08HYNu2bWzbto28vDzuuusuIiMjmTFjBq1btzb6SUpKYtOmTZw+fZq2bdsSHh7OrFmzHJ5DfP311yxbtowvv/yS9u3bM2vWLJYtW8a4ceP4wx/+cMu/u5jP4MGDGTp0KEVFRXz22We4u7szZswYZs6caSxKio+P5/Tp01itVnr16sUf/vAHpk6dylNPPWUswde12PQUZuI0hYWF/O1vfyMsLIw2bdoY7TNnziQ3N5ddu3Y5cXQiYia6zShO4+7uzmuvvcbHH3/MU089RatWrTh8+DDJycksXrzY2cMTERPRzEycKiMjg1WrVvHVV1/x008/0a1bNyZOnMjIkSOdPTQRMRGFmYiImJ6W5ouIiOkpzERExPQUZiIiYnoKMxERMT2FmYiImJ7CTERETO//ARV51kGiiCDiAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#visualizing target_col\n",
    "sns.set_style('darkgrid')\n",
    "sns.barplot(label_counts.index,label_counts.Label)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* **Now that we have the data, we have to vectorize our URLs. I used CountVectorizer and gather words using tokenizer, since there are words in urls that are more important than other words e.g ‘virus’, ‘.exe’ ,’.dat’ etc. Lets convert the URLs into a vector form.**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### RegexpTokenizer\n",
    "* A tokenizer that splits a string using a regular expression, which matches either the tokens or the separators between tokens."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer = RegexpTokenizer(r'[A-Za-z]+')"
   ]
  },
  
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "browser = webdriver.Chrome(r\"chromedriver.exe\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**You can download chromedriver.exe from my github <a href='https://github.com/taruntiwarihp/dataSets/blob/master/chromedriver_win32.zip'>here</a>**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* After set up the Chrome driver create two lists.\n",
    "* First list named list_urls holds all the pages you’d like to scrape.\n",
    "* Second, create an empty list where you’ll append links from each page.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "list_urls = ['https://www.ezeephones.com/','https://www.ezeephones.com/about-us'] #here i take phishing sites \n",
    "links_with_text = []"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* I took some phishing site to see were the hackers redirect(on different link) us.\n",
    "* Use the BeautifulSoup library to extract only relevant hyperlinks for Google, i.e. links only with '<'a'>' tags with href attributes. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### BeautifulSoup\n",
    "* It is use for getting data out of HTML, XML, and other markup languages. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "for url in list_urls:\n",
    "    browser.get(url)\n",
    "    soup = BeautifulSoup(browser.page_source,\"html.parser\")\n",
    "    for line in soup.find_all('a'):\n",
    "        href = line.get('href')\n",
    "        links_with_text.append([url, href])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Turn the URL’s into a Dataframe\n",
    "* After you get the list of your websites with hyperlinks turn them into a Pandas DataFrame with columns “from” (URL where the link resides) and “to” (link destination URL)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame(links_with_text, columns=[\"from\", \"to\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>from</th>\n",
       "      <th>to</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>https://www.ezeephones.com/</td>\n",
       "      <td>None</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>https://www.ezeephones.com/</td>\n",
       "      <td>https://www.ezeephones.com/</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>https://www.ezeephones.com/</td>\n",
       "      <td>/cart</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>https://www.ezeephones.com/</td>\n",
       "      <td>/category/notch-phones</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>https://www.ezeephones.com/</td>\n",
       "      <td>/category/Deals - Of The Day</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                          from                            to\n",
       "0  https://www.ezeephones.com/                          None\n",
       "1  https://www.ezeephones.com/   https://www.ezeephones.com/\n",
       "2  https://www.ezeephones.com/                         /cart\n",
       "3  https://www.ezeephones.com/        /category/notch-phones\n",
       "4  https://www.ezeephones.com/  /category/Deals - Of The Day"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 3: Draw a graph\n",
    "* Finally, use the aforementioned DataFrame to **visualize an internal link structure by feeding it to the Networkx method from_pandas_edgelist first** and draw it by calling nx.draw"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "scrolled": True
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "GA = nx.from_pandas_edgelist(df, source=\"from\", target=\"to\")\n",
    "nx.draw(GA, with_labels=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Creating Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### CountVectorizer\n",
    "* CountVectorizer is used to transform a corpora of text to a vector of term / token counts."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "#create cv object\n",
    "cv = CountVectorizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "#help(CountVectorizer())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature = cv.fit_transform(phish_data.text_sent) #transform all text which we tokenize and stemed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "scrolled": True
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0]], dtype=int64)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "feature[:5].toarray() # convert sparse matrix into array to print transformed features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### * Spliting the data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### LogisticRegression\n",
    "* Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create lr object\n",
    "lr = LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr.fit(trainX,trainY)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9636514559077306"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr.score(testX,testY)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    ".*** Logistic Regression is giving 96% accuracy, Now we will store scores in dict to see which model perform best**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "Scores_ml = {}\n",
    "Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Accuracy : 0.9782480479795345\n",
      "Testing Accuracy : 0.9636514559077306\n",
      "\n",
      "CLASSIFICATION REPORT\n",
      "\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "         Bad       0.90      0.97      0.93     36597\n",
      "        Good       0.99      0.96      0.97    100740\n",
      "\n",
      "    accuracy                           0.96    137337\n",
      "   macro avg       0.95      0.96      0.95    137337\n",
      "weighted avg       0.97      0.96      0.96    137337\n",
      "\n",
      "\n",
      "CONFUSION MATRIX\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1ec84c387c8>"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAW0AAAD4CAYAAAAn3bdmAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deXwV1fnH8c+TALLI7lIkKlBxQUXFDRf8KbaAYotarVi31gW01q2//opYFUWtttpa606liBYRq20FXKhFUKwKgigoqEQFRHAFAVmEJM/vjznBKyY3k5Cb3Jl836/XvHLn3DNzz8Dl4eSZc86YuyMiIslQUN8NEBGR+BS0RUQSREFbRCRBFLRFRBJEQVtEJEEa5foDfj/nGQ1PkW/5dfed67sJkpd2tS09Q7OdTo0dc9YtHrvFn1fX1NMWEUmQnPe0RUTqklm6+6IK2iKSKgWW7rCW7qsTkQZHPW0RkQQxS9y9xWpR0BaRlFFPW0QkMZQeERFJEAVtEZEE0egREZEEUU9bRCRBFLRFRBLE0JA/EZHEUE9bRCRBCgrSHdbSfXUi0gCppy0ikhhKj4iIJIiCtohIgpjSIyIiyaGetohIghQUFNZ3E3JKQVtEUkXpERGRBFF6REQkQRS0RUQSROkREZEEMU1jFxFJDj3YV0QkQZQeERFJEN2IFBFJEqVHREQSJN0dbQVtEUmZgnRHbQVtEUmXdMdsBW0RSRdXTltEJEHSHbMVtEUkZQrSHbVTnv0RkQbHLP5W5ansMjN708zeMLOxZtbUzDqb2XQzW2Bm48ysSai7VdgvDu93yjjP0FD+tpn1zSjvF8qKzezyOJenoC0i6VJo8bcszKwjcDFwgLvvBRQCA4HfAbe6e1dgBXBOOOQcYIW77wLcGuphZt3CcXsC/YC7zKzQzAqBO4FjgG7AqaFuVgraIpIutdjTJkohNzOzRkBzYBnQG3g0vD8aOD68HhD2Ce8fbdFCKAOAh939K3d/HygGDgpbsbu/5+4bgIdD3awUtEUkXSz+ZmaDzGxmxjao/DTu/iFwC7CYKFivBGYBX7h7Sai2BOgYXncEPgjHloT67TPLNzumsvKsdCNSRNKlGjci3X0EMKKi98ysLVHPtzPwBfB3olTGt05Tfkgl71VWXlGn2Sso+wYFbRFJl9obPPI94H13/xTAzP4BHAq0MbNGoTddBCwN9ZcAOwJLQjqlNbA8o7xc5jGVlVdK6RERSRUvLIi9VWEx0NPMmofc9NHAPGAKcFKocxbweHg9PuwT3n/W3T2UDwyjSzoDXYEZwCtA1zAapQnRzcrxVTVKPW0RSZda6mm7+3QzexR4FSgBZhOlUp4AHjaz60PZyHDISOBBMysm6mEPDOd508weIQr4JcCF7l4KYGa/ACYRjUz5q7u/WVW7FLRFJF1qcRq7uw8Dhm1W/B7RyI/N664HTq7kPDcAN1RQ/iTwZHXapKAtIumS8hmRCtoiki7pjtkK2iKSMlrlT0QkQaqYnp50Ctoiki7qaYuIJEi6Y7aCdnWVbNjIE1f/ibKSEspKS+nccz96nNKf5+94kGXzimnSvCkAR1x4Bu07F2067tPiRUy44haOuuxsOh+yHwBPX38nny5YyPa7d6HP0As21XV3Zo2dwPsvz8YKCtijTy/2PPbIOr1O2TJDh97G1Kmv0L59ayZOvBOAp556gTvueIh3313C3//+B/beuysAGzeWcOWVtzNv3ruUlJRy/PG9GTz45ErPI9m5Ro9IpsLGjTh22MU0brYVZSWlTLzqjxTtF62meNAZx28KyJnKSst45W+P03HfPb5R3n3A9yj5agNvPfPCN8oXTH2ZNZ9/wUl/ugorKGDdytW5uyDJiRNPPJrTT+/PkCG3birbddeduf32Kxg27JvB9+mnX2DDho1MmHAH69atp3//C+nf/wiKirav8DxShZSnRzSNvZrMjMbNtgKgrLSUstLSKr8k855+jk4996FZq5bfKN9h7902nSvT/EkvsN9Jx2DhqdLNWrf8Vh3JbwceuBetN/t7++53d6RLl6Jv1TUz1q1bT0lJKevXb6Bx40ZsvXXzSs8jVajGKn9JVGlP28x6ZDvQ3V+t/eYkQ1lpGY8P+R2rPvqUPfodwXZdO/HWpGnMGjuB2Y8+xQ5778aBp/2QwsaNWfP5Fyya/jrHDLuYF4rHxDr/6o8/5b0XZ7Foxus0bdWSnmefROsO2+X4qqS+9O17GJMnT+fww89k/fqvGDr0XNq0UaCusarXFEm0bOmRP4SfTYEDgNeJ/m/qDkwHDq/swLAm7SCAE6+6hINP6l8rjc0XBYUFnHDLUL5as5bJN/+F5YuXcsBpP6RZm1aUlZTwwr1jmfOv/7Dfycfw8v2PceDpAyioxhepdGMJhU0aM+B3Q1g4/TWm3TWG4667LIdXJPVpzpx3KCgoYNq00axa9SU/+cnlHHrovuy443fqu2nJlNAedFyVRhJ3P8rdjwIWAT3c/QB33x/Yj+jJC5Vy9xGh/gFpC9iZtmrRnO/s2ZUPX5tH87atMTMKGzdm16N68mnxQgA+e3cxU/40inE/v5r3X57Ni/eNY+GM17Oet0X7tnQ6eF8Adj5oH5Yv+jDXlyL1aOLE5+jVqweNGzeiffs29OixB3PnLqjvZiVXgcXfEihO9293d59bvuPubwD75q5J+W3dytV8tWYtACVfbWDpnLdp3XF71q5YCUQjPxbNmEPbHXcA4JS7ruWUu4Zzyl3D6dxzPw499xQ6HbRP1s/Y+cDuLHvjHQA+mreA1jsoNZJmHTpsy/Tpc3B31q5dz+uvv11h7ltiSnnQjjN6ZL6Z3Qf8jeipCqcD83Paqjy27otVPHfHg3hZGe5Ol0N6sNP+e/PkNX9m/arVONC+UxGHnTewynNNvOpWVn74MRvXf8XYwVfS64KfULRvN7qf8H2m3jaaNyY+S6OmW3H4+T/J/YVJrfrlL29mxoy5rFixiiOO+CkXXfQT2rRpyXXX3cvy5SsZPHg4e+zRmZEjh3Paaf0ZOvQ2jjvuQtzhxBO/x+67d670PCef3Keery6/eTJjcWwWrdGdpYJZU+AC4IhQ9Dxwd1iGsEq/n/NMlY/PkYbn1913ru8mSF7adYtDbpfBj8WOOe/d+6PEhfgqe9ohON8aNhGR/JbQtEdcVQZtM+sK3Ah0IxpJAoC7d8lhu0REaibdI/5iXd4o4G6ix+QcBTwAPJjLRomI1JhZ/C2B4gTtZu4+mSj/vcjdrwF657ZZIiI1pNEjrDezAmBBeAjlh4DGoIlIXvKE9qDjihO0LwWaAxcD1xH1ss/KeoSISH1p1MCDtru/El5+Cfwst80REdlCKe9pV5rTNrNtzGyYmV1sZlub2d1m9oaZPW5mu9RlI0VEYkt5TjvbjciHgK2ArsAM4D3gJGAicF/umyYiUgMNdWlWYHt3v8LMDFjk7jeH8rfM7MI6aJuISLU15CfXlAK4u5vZZ5u9V5a7JomIbIEGHLS7mNl4ol8iyl8T9jvnvGUiIjVR2HCD9oCM17ds9t7m+yIi+SHlo0cqDdru/lxdNkREpFakPD0Sa2kVM7sm276ISN5I+ZC/ODMiAWZVsS8ikhc0jR1w9wnZ9kVE8kZDvRFpZrcTPV6sQu5+cU5aJCKyJRKa9ogrW097Zp21QkSktjTUoO3uo+uyISIitSLdMTvW48a2BYbw7ceN6UEIIpJ30j6NPc6QvzHAfKJZkNcCC4FXsh0gIlJvavFxY2bWxsweNbO3zGy+mR1iZu3M7BkzWxB+tg11zcz+bGbFZjbHzHpknOesUH+BmZ2VUb6/mc0Nx/w5rPWUVZyg3d7dRwIb3f05dz8b6BnjOBGRuldo8beq3QY87e67A/sQdWAvBya7e1dgctgHOIZoVdSuwCCiZ+tiZu2AYcDBwEHAsPJAH+oMyjiuX1UNihO0N4afy8ysv5ntBxTFOE5EpM4VFMTfsjGzVsARwEgAd9/g7l8QLfFRfs9vNHB8eD0AeMAjLwNtzKwD0Bd4xt2Xu/sK4BmgX3ivlbu/5O5O9ND08nNVKs447evNrDXwv8DtQCvgshjHiYjUuerMrTGzQUQ93XIj3H1EeN0F+BQYZWb7EE0qvIRo2eplAO6+zMzKn5nbEfgg41xLQlm28iUVlGcV53FjE8PLlcBRVdUXEalP1QnaIUCPqOTtRkAP4CJ3n25mt/F1KqTCj67oI2pQnlWc0SOjKjpRyG2LiOSVGPfy4loCLHH36WH/UaKg/bGZdQi97A7AJxn1d8w4vghYGsqP3Kx8aigvqqB+VnFy2hOBJ8I2mSg98mWM40RE6lxt5bTd/SPgAzPbLRQdDcwDxgPlI0DOAh4Pr8cDZ4ZRJD2BlSGNMgnoY2Ztww3IPsCk8N5qM+sZRo2cmXGuSsVJjzyWuW9mY4H/VHWciEh9sFhrl8Z2ETDGzJoQPSf3Z0Sd3UfM7BxgMXByqPskcCxQDKwNdXH35WZ2HV8PlR7u7svD6wuA+4FmwFNhyyruKn+ZugI71eA4EZGcq81F/tz9NeCACt46uoK6DlT4/Fx3/yvw1wrKZwJ7VadNcXLaq/lmTvsjohmSIiJ5J+UTImOlR1rWRUNERGpDypfTrvpGpJlNjlMmIpIPanEWe17Ktp52U6A5sE2441l+ia2AHeqgbSIi1VbQUB+CAAwGLiUK0LP4OmivAu7McbtERGokqT3ouLKtp30bcJuZXeTut9dhm0REaiztQTvOiMYyM2tTvhMGiP88h20SEamxtOe04wTt88LKVgCEVarOy12TRERqrsDib0kUZ3JNgZlZGDiOmRUCTXLbLBGRmklqDzquOEF7EtGUzXuIJtmcT4ypliIi9aEhjx4pN4RovdkLiEaQzAY65LJRIiI1lfaedpU5bXcvA14mWizlAKI59/Nz3C4RkRpJ+43IbJNrdgUGAqcCnwPjANxdD0IQkbyV1GAcV7b0yFvANOAH7l4MYGZ6zJiI5LWkjgqJK1t65EdEK/pNMbO/mNnRVPx4HBGRvFFQGH9LokqDtrv/091PAXYnejTOZcD2Zna3mfWpo/aJiFRL2nPacW5ErnH3Me5+HNEzzF4j+8MtRUTqjZnF3pKoWg/mcffl7n6vu/fOVYNERLZE2nvaNXncmIhI3kpqMI4r50H7l3ttl+uPkARqttOw+m6C5KF1i8du8TkUtEVEEqRR7T6NPe8oaItIqhSYV10pwRS0RSRV0j65RkFbRFIl5dkRBW0RSRelR0REEkTpERGRBGmkoC0ikhym9IiISHIoPSIikiAaPSIikiAaPSIikiC6ESkikiDKaYuIJIjSIyIiCaKetohIgqR99Ejar09EGpgC89hbHGZWaGazzWxi2O9sZtPNbIGZjTOzJqF8q7BfHN7vlHGOoaH8bTPrm1HeL5QVm1msZ+8qaItIqjQqiL/FdAkwP2P/d8Ct7t4VWAGcE8rPAVa4+y7AraEeZtYNGAjsCfQD7gr/ERQCdwLHAN2AU0PdrBS0RSRVCqqxVcXMioD+wH1h34DewKOhymjg+PB6QNgnvH90qD8AeNjdv3L394Fi4KCwFbv7e+6+AXg41K3y+kREUqOW0yN/An4NlIX99sAX7l4S9pcAHcPrjsAHAOH9laH+pvLNjqmsPPv1xWm1iEhSFFj8zcwGmdnMjG1Q+XnM7DjgE3eflXH6isameBXvVbc8K40eEZFUqU5P1N1HACMqefsw4IdmdizQFGhF1PNuY2aNQm+6CFga6i8BdgSWmFkjoDWwPKO8XOYxlZVXSj1tEUmV6vS0s3H3oe5e5O6diG4kPuvupwFTgJNCtbOAx8Pr8WGf8P6z7u6hfGAYXdIZ6ArMAF4BuobRKE3CZ4yv6vrU0xaRVCksyPmMyCHAw2Z2PTAbGBnKRwIPmlkxUQ97IIC7v2lmjwDzgBLgQncvBTCzXwCTgELgr+7+ZlUfrqAtIqmSi/SBu08FpobX7xGN/Ni8znrg5EqOvwG4oYLyJ4Enq9MWBW0RSRWtPSIikiBae0REJEEUtEVEEqSx0iMiIsmhnraISIIoaIuIJEihgraISHKopy0ikiAapy0ikiCN1dMWEUkOpUdERBJE6RERkQTR6BERkQRRekREJEGq8ZT1RFLQFpFUKVROW0QkOVLe0VbQFpF0UU5bRCRBFLRFRBJEOW0RkQTR6BERkQRRekREJEE0I1JEJEG09ohU6quvNnDmGcPYsKGE0pJS+vTtyS8u+jEvvzSXW27+G2VeRvPmTbnhtxey887fYeYr87jpxtG8884ibv7DpfTt23PTuQaddwNzXl9Ajx67c9c9l9fjVUlNXXh2P352am/MjFFjn+WOkU8BcMFP+3L+WX0oKS3j6Wdn85vfPkTjxoXcceO59OjehbIy51fXjGbay/Np1rQJY+6+lC47b0dpmfPkf2Zx1U0PA7BTx22455bBbNOuFSu++JKzL7mTDz9aXp+XnJdSntJW0N4STZo05q+jhtGiRVM2bizhjNOvplevfRl+7X3cfuf/8d3vFjH2oUnce89j/PbGC+mwwzbccOPPuf+vE751rrPP/iHr1n/F38f9px6uRLZUt12L+Nmpven1gyvZsLGE8Q9ezlOTZ9OxQzuO67M/B/YdwoYNJWzbvhUAZ5/aG4AD+wxh2/at+NcDQzj8uCsB+NOIiTz/0jwaNy7kqbFX0ufIffj31Ne58crTGPPYNMY8+jz/c+ieDL98IOdcele9XXO+SntOO+3/KeWUmdGiRVMASkpKKdlYiplhBmu+XAfAl1+uZbvt2gLQseN27LbbzlgF36qeh+xNixbN6q7xUqt279qRGa8uYN36DZSWljHt5fkM6Hcgg874PrfcNZ4NG0oA+PTzVaF+EVP+++amspWr1rJ/9y6sW7+B51+aB8DGjaW89sb7dOzQftMxU194A4DnXnyT476/f11fZiI0LvDYWxIpaG+h0tIyTjzh/+h1+LkccujedN+nK8OvO5/zB99I7yPPZ/z45zn3vOPru5mSY2++/QGHH7wH7dpsTbOmTeh31L4UdWjPLp2/w2EH7c7zj1/Hvx+5mv27dwFg7vxF/KDP/hQWFrDzjtuy316dKdqh/TfO2bpVc479Xg+m/DcK1HPnLeL4Yw8CYEC/A2nVsjnt2mxdtxeaAAUWf0uiSoO2ma02s1WVbdlOamaDzGymmc38y4hHa7/VeaSwsIB//PNmnp1yD3PnvsuCdxbzwOgnuOfeoTw79R5OOOEofn/TA/XdTMmxt4uX8oe7xzNxzBWMf/By5sxfTElpKY0aFdK2dQuOGHAVV9wwhr/ddQkAo8dN5cNly/nvxBu4ediZvDzrHUpKSjedr7CwgNG3X8RdoyaxcPEnAAy9YQy9Dt6Dl568kV499+DDZZ9TUlpaYXsasrQH7Upz2u7eEsDMhgMfAQ8CBpwGtMx2UncfAYwAKCl7PZm/g1RTq1YtOOigbkyb9hpvv72I7vt0BaDfMYcyeNAN9dw6qQujx01l9LipAFz761P4cNlydtulI/96agYAM19/lzJ3tmnXks+Wr+bXwx/cdOyUf1xL8cKPNu3fedN5vLvwo003MwGWfbyCgYNvBaBF8604/piDWLV6XR1cWbKkPX0Q5/r6uvtd7r7a3Ve5+93Aj3LdsCRYvnwVq1atAWD9+g289NJcunTpyOrVa1n4/lIAXnpxDl26dKzPZkodKb/JuOMO7RnQ70AeGf8iE/49kyMP3ROAXTp/hyaNG/HZ8tU0a9qE5s22AqB3r70pKS3lrQUfAjDsVz+mdctm/Oqab/6G1r5tS8yi7uH/XThg038Q8k1m8bckijN6pNTMTgMeBhw4FdDvZMCnn67giqF3UlZaRlmZ07ffIRx51P5cO3wwl17yB6yggNatWnDdDRcAMHduMZdcdAurVq1h6pRZ3Hn7I4yf+EcAzjj9at5/70PWrl1P7yPPZ/j153P44fvW5+VJNY299zLatd2ajRtLufSqUXyxcg2jx03h3pvPZ+Yzv2fDhhLO/eXdAGy7TSsmPDiUsjJn6cfLN40C6fiddlx+8Qm8teBDXnrytwDcM/rf3P/wFI44ZA+GDxmIO7wwfT6XXjWq3q41nyU17RGXuWfPXphZJ+A24LBQ9AJwqbsvjPMBDSU9ItXTstNN9d0EyUPrFo/d4pD76mdPxI45Pbbpn7gQX2VPOwTnAblviojIlrOUz4isMqdtZkVm9k8z+8TMPjazx8ysqC4aJyJSXVaNLYni3IgcBYwHdgA6AhNCmYhI3kn7jcg4QXtbdx/l7iVhux/YNsftEhGpEfW04TMzO93MCsN2OvB5rhsmIlIThRZ/y8bMdjSzKWY238zeNLNLQnk7M3vGzBaEn21DuZnZn82s2MzmmFmPjHOdFeovMLOzMsr3N7O54Zg/m1Xd/48TtM8Gfkw0weYj4KRQJiKSd2oxPVIC/K+77wH0BC40s27A5cBkd+8KTA77AMcAXcM2CLg7ao+1A4YBBwMHAcPKA32oMyjjuH5VNSrO6JHFwA+rvDwRkTxQW2kPd18GLAuvV5vZfKL7egOAI0O10cBUYEgof8CjcdQvm1kbM+sQ6j7j7ssBzOwZoJ+ZTQVauftLofwB4Hjg62mwFdDoERFJlerktDPXSQrboArPGc1X2Q+YDmwfAnp5YN8uVOsIfJBx2JJQlq18SQXlWcWZETkKeAg4OeyfHsq+H+NYEZE6VZ0ZkZnrJFXGzLYGHiOaVLgqS9q5oje8BuVZafSIiKRKbY4eMbPGRAF7jLv/IxR/HNIehJ+fhPIlwI4ZhxcBS6soL6qgPCuNHhGRVCkwj71lE0ZyjATmu/sfM94aD5SPADkLeDyj/MwwiqQnsDKkTyYBfcysbbgB2QeYFN5bbWY9w2edmXGuSsVJj5wN3AHcStR1fxGNHhGRPFWLk2YOA84A5prZa6HsCuAm4BEzOwdYzNep4yeBY4FiYC3wMwB3X25m1wGvhHrDy29KAhcA9wPNiG5AZr0JCTEWjNpSWjBKKqIFo6QitbFg1MLVE2LHnE4tf5C4OTaV9rTNbE/gu+4+PuzfCrQOb9/h7q/WQftERKolqdPT48qW074J+Cxjvy/wBDAFuDqXjRIRqam0T2PPltPu4O4vZuyvcvfHAMxscG6bJSJSM2l/CEK2oP2N50C6e8+M3e0QEclDaQ/a2dIjS83s4M0Lw1CWKscSiojUh4acHhkCjDOz+4Hym477E41LPCXH7RIRqZEG++Qad59BtCpVIfDTsBUAPcN7IiJ5pyH3tHH3T9BIERFJkIY85G8TM7sm276ISL4orMaWRHGmsQPMqmJfRCQvpL2nHStou/uEbPsiIvkj3VE72zT228mytqu7X5yTFomIbAFrqEEbmFlnrRARqSVmsW7VJValQdvdR9dlQ0REakfD7WkDYGbbEk206QY0LS939945bJeISI1YvEFxiRXn6sYA84HOwLXAQr5ezFtEJK+YFcTekihOq9u7+0hgo7s/5+5nAz2rOkhEpH6ke05knCF/G8PPZWbWn2ixqKIs9UVE6k1DHj1S7nozaw38L3A70Aq4LKetEhGpoQYftN19Yni5Ejgqt80REdkyZkmdoB5PnNEjo6hgkk3IbYuI5JkG3tMGJma8bgqcgB6CICJ5SumR8FzIcmY2FvhPzlokIrJFkjmUL664q/xl6grsVNsNERGpDQ2+p21mq/lmTvsjohmSIiJ5x1K+Nmuc9EjLquqIiOQLS+zjDeKpMvljZpPjlImI5IcGOiPSzJoCzYFtzKwtX19hK2CHOmibiEi1NeT0yGDgUqIAPYuvg/Yq4M4ct0tEpIYaaNB299uA28zsIne/vQ7bJCJSY1qaFcrMrE35jpm1NbOf57BNIiJbIN057ThB+zx3/6J8x91XAOflrkkiIjVXYAWxtySKM7mmwMzM3R3AotVYmuS2WSIiNZXMYBxXnKA9CXjEzO4hmmRzPvBUTlslIlJDDX5GJNHsx0HABURJoNlAh1w2SkSk5tIdtKv8PcLdy4CXgfeAA4CjiZ4ZKSKSd8ws9pZE2SbX7AoMBE4FPgfGAbi7HoQgInkr7dPYLdxf/PYbZmXANOAcdy8OZe+5e5c6bF+qmNkgdx9R3+2Q/KLvhVRHtvTIj4hW9JtiZn8xs6NJe7Io9wbVdwMkL+l7IbFVGrTd/Z/ufgqwOzCV6GG+25vZ3WbWp47aJyIiGeLciFzj7mPc/TigCHgNuDznLRMRkW+p1ih0d1/u7ve6e+9cNSjllLeUiuh7IbFVeiNSRETyT7rne4qIpIyCtohIgjSIoG1mpWb2mpm9YWZ/N7PmW3Cu+83spPD6PjPrlqXukWZ2aA0+Y6GZbVNB+VQzeztcy3wzq9ZQsdCeidVtT5qk6LvQyMx+a2YLwvW8Zma/qe75K/nMa8zsV7VxLql9DSJoA+vcfV933wvYQLTo1SZh5cJqc/dz3X1elipHAtX+h1qF09x9X+Aw4HdmphUXqyct34XriZ4qtXf4PvQCGtfi+SVPNZSgnWkasEvo+Uwxs4eAuWZWaGY3m9krZjbHzAYDWOQOM5tnZk8A25WfKPR8Dwiv+5nZq2b2uplNNrNORAHhstAL6mVm25rZY+EzXjGzw8Kx7c3s32Y228zuJd4kpq2BNUBpOMfdZjbTzN40s2sz2tjPzN4ysxeAE7f8jy9VEvldCL8dnAdc5O7rAdx9tbtfk1Hnl+G3iTfM7NIY5b8Jv8X9B9ittv6AJQfcPfUb8GX42Qh4nGjFwiOJgl7n8N4g4MrweitgJtCZKNA9AxQS9Wy+AE4K9aYSLaK1LfBBxrnahZ/XAL/KaMdDwOHh9U7A/PD6z8DV4XV/oiVwtwn7TwI7ZHze28AcYB0wOOPc5Z9ZGOp1B5qGdnUl+sf/CDCxvv8+9F3Ysu9C+LudneUa9wfmAi2I/nN/E9gvRnlzogd3F2e2VVt+bXGWZk2DZmb2Wng9DRhJ9KvqDHd/P5T3AbqX5yiB1kTB7ghgrLuXAkvN7NkKzt8TeL78XO6+vJJ2fA/oZl+vLtbKzFqGzzgxHPuEma0or+Dux252jtPcfaaZbQu8aGZPu/si4Mchx92IaOncbkS/Sb3v7gsAzOxvaAImFYIAAAHzSURBVMp04r8Lm+e4zexnwCVA+3AthwP/dPc14f1/EKVPrJLyglC+NpSPr6TNkgcaStBe51Heb5Pwj2VNZhHRr5uTNqt3LFFvJxuLUQeifxyHuPu6CtpSrQHz7v6pmb0KHGxmBcCvgAPdfYWZ3U/Uy672eRuANHwXioGdzKylR2mRUcAoM3uD6LeAytJr2dJu+p4kREPMaVdmEnCBmTWGaGlaM2sBPA8MDHnODkBFS9O+BPyPmXUOx7YL5auBlhn1/g38onzHzMqDx/PAaaHsGKBtVY0Nec39gHeJfqVdA6w0s+2BY0K1t4DOZvbdsH9qVecVIM+/C6FHPBK4w8yahrqZjwF8HjjezJqHdp9A9FtFtvITzKxZ6O3/IOafk9SDhtLTjuM+oBPwqkXdnU+B44F/Ar2Jcn7vAM9tfmDo9Q4C/hF6vZ8A3wcmAI+a2QDgIuBi4E4zm0P0Z/880Q2qa4Gxoef8HLC4/Nxm9iRwrrsvDUVjzGwdUa71fnefFerNJspRvgf8N7RrfWjXE2b2GfACsFct/FmlXRK+C78BrgPeMLPVRPc4RgNL3X1h+G1rRvn1uPvscI7KyscRrSu0iCiQS57SNHYRkQRRekREJEEUtEVEEkRBW0QkQRS0RUQSREFbRCRBFLRFRBJEQVtEJEH+H8R9o7Qt9WjxAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "print('Training Accuracy :',lr.score(trainX,trainY))\n",
    "print('Testing Accuracy :',lr.score(testX,testY))\n",
    "con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),\n",
    "            columns = ['Predicted:Bad', 'Predicted:Good'],\n",
    "            index = ['Actual:Bad', 'Actual:Good'])\n",
    "\n",
    "\n",
    "print('\\nCLASSIFICATION REPORT\\n')\n",
    "print(classification_report(lr.predict(testX), testY,\n",
    "                            target_names =['Bad','Good']))\n",
    "\n",
    "print('\\nCONFUSION MATRIX')\n",
    "plt.figure(figsize= (6,4))\n",
    "sns.heatmap(con_mat, annot = True,fmt='d',cmap=\"YlGnBu\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### MultinomialNB\n",
    "* Applying Multinomial Naive Bayes to NLP Problems. Naive Bayes Classifier Algorithm is a family of probabilistic algorithms based on applying Bayes' theorem with the “naive” assumption of conditional independence between every pair of a feature."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create mnb object\n",
    "mnb = MultinomialNB()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MultinomialNB()"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mnb.fit(trainX,trainY)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9574550194048217"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mnb.score(testX,testY)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*** MultinomialNB gives us 95% accuracy**  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Protections\n",
    "#### How to Protect Your Computer \n",
    "Below are some key steps to protecting your computer from intrusion:\n",
    "\n",
    "1. **Keep Your Firewall Turned On:** A firewall helps protect your computer from hackers who might try to gain access to crash it, delete information, or even steal passwords or other sensitive information. Software firewalls are widely recommended for single computers. The software is prepackaged on some operating systems or can be purchased for individual computers. For multiple networked computers, hardware routers typically provide firewall protection.\n",
    "\n",
    "2. **Install or Update Your Antivirus Software:** Antivirus software is designed to prevent malicious software programs from embedding on your computer. If it detects malicious code, like a virus or a worm, it works to disarm or remove it. Viruses can infect computers without users’ knowledge. Most types of antivirus software can be set up to update automatically.\n",
    "\n",
    "3. **Install or Update Your Antispyware Technology:** Spyware is just what it sounds like—software that is surreptitiously installed on your computer to let others peer into your activities on the computer. Some spyware collects information about you without your consent or produces unwanted pop-up ads on your web browser. Some operating systems offer free spyware protection, and inexpensive software is readily available for download on the Internet or at your local computer store. Be wary of ads on the Internet offering downloadable antispyware—in some cases these products may be fake and may actually contain spyware or other malicious code. It’s like buying groceries—shop where you trust.\n",
    "\n",
    "4. **Keep Your Operating System Up to Date:** Computer operating systems are periodically updated to stay in tune with technology requirements and to fix security holes. Be sure to install the updates to ensure your computer has the latest protection.\n",
    "\n",
    "5. **Be Careful What You Download:** Carelessly downloading e-mail attachments can circumvent even the most vigilant anti-virus software. Never open an e-mail attachment from someone you don’t know, and be wary of forwarded attachments from people you do know. They may have unwittingly advanced malicious code.\n",
    "\n",
    "6. **Turn Off Your Computer:** With the growth of high-speed Internet connections, many opt to leave their computers on and ready for action. The downside is that being “always on” renders computers more susceptible. Beyond firewall protection, which is designed to fend off unwanted attacks, turning the computer off effectively severs an attacker’s connection—be it spyware or a botnet that employs your computer’s resources to reach out to other unwitting users."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "https://research.aalto.fi/en/datasets/phishstorm-phishing-legitimate-url-dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.8.0"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": False
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
