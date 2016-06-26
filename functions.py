import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
from time import time

'''
text preprocessing function
  - tokenize
  - remove punctuation
  - remove stopwords, empty strings, nonsense java strings
'''

def preprocess_article_content(text_df):

    print 'preprocessing article text...'
    
    # text_df is data frame from SQL query, column 'content' contains text content from each article
    article_list = []
    
    # define punctuation to remove
    punc=set('''`~!@#$%^&*()-_=+\|]}[{;:'",<.>/?''')
    
    tokenizer = WhitespaceTokenizer()
    stop_words = set(stopwords.words('english'))
    #stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    
    kept_rows = []
    
    for row, article in enumerate(text_df['content']):
        
        cleaned_tokens = []
        
        tokens = tokenizer.tokenize(article.decode('unicode-escape', 'ignore').lower())
        
        for token in tokens:
            token = ''.join(ch for ch in token if ch not in punc)
            
            if token not in stop_words:
                
                if len(token) > 0 and len(token) < 20: 
                    
                    if not token[0].isdigit() and  not token[-1].isdigit(): 
                        #stemmed_token = stemmer.stem(token)
                        lemmatized_tokens = lemmatizer.lemmatize(token)
                        #cleaned_tokens.append(stemmed_token)
                        cleaned_tokens.append(lemmatized_tokens)
        
        # join cleaned tokens into a string for subsequent LDA
        # filtering out content that is likely noise (error messages etc)
        if len(cleaned_tokens) > 100:
            article_list.append(' '.join(wd for wd in cleaned_tokens))
            kept_rows.append(row)

    print 'preprocessed content for %d articles' % len(article_list)
        
    return article_list, kept_rows


'''
define function to print top words from LDA output
'''

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        line = []
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            line.append([feature_names[i],model.components_[topic_idx,i]])
        print line
    print()



'''
function for processing - vectorizing - and applying DLA

'''

def process_vect_lda(df, vectorizer, lda):
    t0 = time()
    print 'preprocessing article text...'
    df_processed, kept_rows = preprocess_article_content(df)
    usable_df = df.iloc[kept_rows].reset_index(drop = True)
    print 'finished processing in %0.3fs' % (time()-t0)
    t1 = time()
    print 'applying vectorizer to text...'
    vectorized_text = vectorizer.transform(df_processed)
    print 'finished vectorizing in %0.3fs' % (time() - t1)
    t2 = time()
    print 'applying LDA model to text....'
    
    topic_distribution = lda.transform(vectorized_text) # keep as np.array for now
    print 'finished LDA in %0.3fs' % (time() - t2)
    #print 'combining data frame with lda topic distribution...'
    #df_combo = pd.concat([df.reset_index(drop = True), topic_distribution], axis = 1)
    
    print 'finished all in %0.3fs' % (time() - t0)
    
    return usable_df, topic_distribution, df_processed, vectorized_text

