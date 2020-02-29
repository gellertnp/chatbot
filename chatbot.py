# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens

import re
import numpy as np
from PorterStemmer import PorterStemmer

posMovieResp = ["I'm glad you liked %s.", "Yeah, %s is a great movie!",  "I liked %s too!", "%s is a great choice.", "%s is my mom's fav!", "I CRIED when I saw %s."]
negMovieResp = ["I'm so sorry you didn't like %s!", "Yeah, you're not the only one who didn't like %s...", "Oof, ok, I won't recommend movies like %s", "Good to know, but don't talk to my mom, she loved %s.", "%s goes on our no-show list, noted!"]
canRec = [" You've named enough movies for a recommendation, would you like one?", " Woo! Thanks for all the movie ratings, do you want a recommendation now?", " I like your style!! Can I recommend a new movie?", " You've got good taste! Want a recommendation?"]
cantRec = [" Keep rating movies for a recommendation!", " I need a couple more for a rcommendation, let's hear about some more!", " You're on a roll! How about another?", " Keep it up!", " You got any more?", " What about another movie you HATED?", " What about another movie you LOVED?"]
arbitraryResp = ["Please name a movie!", "Sorry, I'm designed to recommend a movie, let's talk about that!", "I'd love to discuss movies, could you tell me about one you've seen?", "Let's talk about movies, please!", "I'd love to talk about that, but I'm paid to give you movie recommendations, could you name a movie please?"]
neutMoviesResp = ["Sorry, could you let me know how you felt about %s", "I didn't catch that. How did you feel about %s", "Oops, I missed your reaction, how did you feel about %s", "Yikes, something's off! How did you feel about %s", "You mentioned %s, but not whether you liked it! How did you feel about it?"]
yeses = ["yes", "yeah", "yep", "ya", "yea", "uh huh", "yas", "mhm", "ye", "mhmm", "mmhmm", "yeet", "mmhm", "mm hm"]
nos = ["no", "nope", "nah", "negative", "nuh uh", "noo", "naw", "mm mm"]

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'Marseille the Shell'

        self.creative = creative
        #holds indeces and sentiment
        self.lastAmbiguous = ["", 0]
        self.flags = {"LastSentiment": False, "LastMovie": []}
        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = movielens.ratings()
        self.sentiment = movielens.sentiment()
        self.toRec = False #Whether a recommendation is ready
        self.title_names = [i[0].lower() for i in self.titles]
        self.userRatings = np.zeros(len(self.title_names))
        self.userSentiments = {}

#        self.p = PorterStemmer()
        #############################################################################
        # TODO: Binarize the movie ratings matrix.
        # @ Max
        ratings = self.binarize(ratings, 2.5)
        #############################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = ratings
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        #############################################################################
        # TODO: Write a short greeting message                                      #
        # @Kayla
        #############################################################################

        greeting_message = "Hello! Are you a human searching for a couple hours of \
        entertainment and want an opinion from a lovely but admittedly idiotic \
        chatbot? If so, then you're in the right place! How can I best aid your \
        decision making process?"

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return greeting_message

    def goodbye(self):
        """Return a message that the chatbot uses to bid farewell to the user."""
        #############################################################################
        # TODO: Write a short farewell message                                      #
        # @Kayla
        #############################################################################

        goodbye_message = "Hope your intended entertainment plans work out \
        and your friends don't flake! Have a nice life."

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return goodbye_message

    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        #############################################################################
        # TODO: Implement the extraction and transformation in this method,         #
        # possibly calling other functions. Although modular code is not graded,    #
        # it is highly recommended.                                                 #
        #############################################################################
        
        movie = ""


        input = self.preprocess(line)
        if self.toRec == True:

            for y in yeses:
                if y in input.lower():
                    self.toRec = False
                    curRecs = self.recommend(self.userRatings, self.ratings)
                    return "How about " + self.titles[curRecs[0]] + "?"
            for n in nos:
                if n in input.lower():
                    return "Ok, want to name another movie?"
            return "Sorry, could you be a little clearer?"

        response = ""
        curMovies = []
        sentiment = ""
        #disambiguating responses
        if self.lastAmbiguous[0] != "":
            movies = [self.lastAmbiguous[0]]
            curMovies.append(self.disambiguate(line, self.lastAmbiguous[0]))
            sentiment = self.get_sentiment_words(self.lastAmbiguous[1])
            self.lastAmbiguous = ["", 0]
            
        elif self.flags["LastSentiment"]:
            movies = self.flags["LastMovie"]
            curMovies = self.flags["LastMovie"]
            self.flags["LastMovie"] = []
            self.flags["LastSentiment"] = False

        else:
            movies = self.extract_titles(input)
            for m in movies:
                curMovies.append(self.find_movies_by_title(m))

        #can't find a movie
        if len(movies) == 0:
            response = np.random.choice(arbitraryResp)

        #getting new sentiment
        if sentiment == "":
            sentiment = self.get_sentiment_words(self.extract_sentiment(line))


        #More than one listed movie, this could be edited for style
        if len(movies) > 1:
            sentiment = self.extract_sentiment_for_movies(input)
            self.lastAmbiguous = ["", 0]
            for i in range(len(curMovies)):
                if len(curMovies[i]) < 1:
                    response += "I couldn't find any movies called " + movies[i] + "."
                if len(curMovies[i]) > 1:
                    response += self.list_ambiguity(curMovies[i], sentiment[i][1])
                else:
                    sentiments = self.get_sentiment_words(sentiment[i][1])
                    movie = self.titles[curMovies[i][0]][0]
#                    movie = self.title_names[curMovies[i][0]][0]
                    response += "You " + sentiments + movie +". "
                    if sentiments == " saw":
                        self.flags["LastSentiment"] = True
                        self.flags["LastMovie"].append(curMovies[i])
                        response += "How did you feel about " + movie + "?"

            #one movie
        else:
            #Must disambiguate
            if len(curMovies[0]) > 1:
                return self.list_ambiguity(curMovies[0], self.extract_sentiment(line))

            #if clarified, call disambiguate


            #If not ambiguous
            if len(curMovies[0]) < 1:
                return "I couldn't find any movies called " + movies[0] + "."
            self.lastAmbiguous = ["", 0]
            self.userRatings[curMovies[0][0]] = self.extract_sentiment(line)
            movie = cur[0]
            if sentiment == "liked ":
                response = np.random.choice(posMovieResp) % movie
            elif sentiment == "saw ":
                response = np.random.choice(neutMoviesResp) %movie
                self.flags["LastSentiment"] = True
                self.flags["LastMovie"].append(curMovies[0])
            else:
                response = np.random.choice(negMovieResp) % movie
        if len(self.userSentiments) > 5:
            self.toRec = True
            response += np.random.choice(canRec)
        else:
            response += np.random.choice(cantRec)


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return response

    def get_sentiment_words(self, integer):
        if integer> 0:
            sentiment = "liked "
        elif integer<0:
            sentiment = "didn't like "
        else:
            sentiment = "saw "
        return sentiment

    def list_ambiguity(self, potential_titles, sentiment):
        response = "I have a couple movies with that name! Could you clarify?: "
        self.lastAmbiguous = [potential_titles, sentiment]
        for i in potential_titles:
            response += self.titles[i][0] + ", "
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information from a line of text.

        Given an input line of text, this method should do any general pre-processing and return the
        pre-processed string. The outputs of this method will be used as inputs (instead of the original
        raw text) for the extract_titles, extract_sentiment, and extract_sentiment_for_movies methods.

        Note that this method is intentially made static, as you shouldn't need to use any
        attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        #############################################################################
        # TODO: Preprocess the text into a desired format.                          #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to your    #
        # implementation to do any generic preprocessing, feel free to leave this   #
        # method unmodified.                                                        #
        #############################################################################
        # articles = [" a ", " an ", " the "]
        # for a in articles:
        #     text = text.replace(a, " ")
        # caps = ["A ", "The ", "An "]
        # for a in caps:
        #     text = text.replace(a, "")
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess('I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text

        @Ella
        """
        titles = []

        if (preprocessed_input.find("\"") != -1):

            # if the doc contains quotations
            start = preprocessed_input.find("\"")
            while start != -1:
                end = preprocessed_input.find("\"", start+1)
                title = preprocessed_input[start+1:end]
                titles.append(str(title))
                start = preprocessed_input.find("\"", end+1)

        else :
            # if doc does not contain quotations
            feelingwords  = ["think", "thought", "felt that", "enjoy", "enjoyed", "like", "hate", "hated"]
            endwords = ["was", "is", "has", "\.", "\!", "\,"]
            for word in feelingwords:
                firstletter = preprocessed_input.find(word)
                if firstletter != -1:
                    start = firstletter + len(word)
                    for endW in endwords :
                        end = preprocessed_input.find(endW)
                        if end != -1:
                            title = preprocessed_input[start+1: end-1]
                            titles.append(str(title.lower()))
        return titles

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 1953]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies

        @Kayla Ella
        """


        # return [i for i in self.title_names if i.find(title) != -1]
        title = title.lower()
        alternate = self.move_start_article(title)
        return [indx for indx, i in enumerate(self.title_names) if (i.find(title,0) != -1 or i.find(alternate, 0) != -1)]
    
    def move_start_article(self, line):
        articles = ['an', 'a', 'the', 'le', 'la', 'les', 'los', 'las', 'el', 'die', 'der', 'das', 'un', 'une', 'des', 'una', 'uno', 'il', 'gle', 'ein', 'eine']
        for a in articles:
            i = line.lower().find(a + ' ')
            if i == 0:
                if line.find('(') != -1:
                    processed = line[i + len(a) + 1:line.find('(')-1] + ', ' + line[:i+len(a)] + line[line.find('(') - 1:]
                else:
                    processed = line[i + len(a) + 1:] + ', ' + line[:i+len(a)]
                return processed.lower()
        return line

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the text
        is super negative and +2 if the sentiment of the text is super positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess('I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text

        @Ella
        """
        
      # TODO: add -2/2 weighting, NEGATIONS
        sentiment = 0

        #TODO: split words and remove movie titles
        # titles = self.extract_titles(preprocessed_input)
        # # print(titles)
        # for t in titles:
        #     preprocessed_input = preprocessed_input.replace(t, '')
        # print("STRING", preprocessed_input)
        # print(self.sentiment.keys())
        negate = 1
        for w in preprocessed_input.split():
            # print("HI", w, w in self.sentiment)
            # if preprocessed_input = " but not ":
            #     print("WORD", w, self.p.stem(w, 0, len(w)-1))
            if w not in self.sentiment:
                w = self.p.stem(w, 0, len(w)-1)
            if w in self.sentiment:
                senti = self.sentiment[w]
                if senti == 'pos':
                    sentiment += negate*1
                elif senti == 'neg':
                    sentiment += negate*-1
                # print("HI", w, self.sentiment[w])
                negate = 1
            elif w == 'not' or w.find('n\'t') != -1:
                negate = -1

            #TODO
        # print("HELLO",sentiment)
        if sentiment == 0 and negate == -1: return -1
        if sentiment == 0: return 0
        return -1 if sentiment < 0 else 1


    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of pre-processed text
        that may contain multiple movies. Note that the sentiments toward
        the movies may be different.

        You should use the same sentiment values as extract_sentiment, described above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess('I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie title,
          and the second is the sentiment in the text toward that movie

        @ Ella
        """


        extracted = []
        titles = self.extract_titles(preprocessed_input)
        input = preprocessed_input
        prev_senti = 0
        for t in titles:
            end = input.find(t)
            sentiment = self.extract_sentiment(input[:end])

            if sentiment == 0 and len(input[:end].replace(' ', '').split()) == 1:
                # print("HEY", prev_senti)
                sentiment = prev_senti
            input = input[end + len(t):]
            # print(input)
            extracted.append((t, sentiment))
            prev_senti = sentiment
        return extracted

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least edit distance
        from the provided title, and with edit distance at most max_distance.

        - If no movies have titles within max_distance of the provided title, return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given title
          than all other movies, return a 1-element list containing its index.
        - If there is a tie for closest movie, return a list with the indices of all movies
          tying for minimum edit distance to the given movie.

        Example:
          chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance

        @ Max
        """
        movies = []
        title = title.lower()
        length = len(title)
        for candidate in self.title_names:
            index = self.title_names.index(candidate)
            candidate = candidate[:candidate.find(' (')]
            if np.absolute(len(candidate) - length) <= max_distance:
                dist = self.get_edit_distance(candidate, title)
                if dist <= max_distance:
                    if dist < max_distance:
                        movies = []
                        max_distance = dist
                    movies.append(index)
        return movies

    def get_edit_distance(self, word1, word2):
        """helper method for find_movies_closest_to_title"""
        distances = np.zeros((len(word1) + 1, len(word2) + 1))
        for i in range (len(word1) + 1):
            distances[i][0] = i
        for i in range (len(word2) + 1):
            distances[0][i] = i
        for i in range (1,len(word1) + 1):
            for j in range (1,len(word2) + 1):
                diag = 2
                if word1[i - 1] == word2[j - 1]:
                    diag = 0
                distances[i][j] = min(distances[i-1][j] + 1, distances[i][j-1] + 1, distances[i-1][j-1] + diag)
        return distances[len(word1)][len(word2)]

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be talking about
        (represented as indices), and a string given by the user as clarification
        (eg. in response to your bot saying "Which movie did you mean: Titanic (1953)
        or Titanic (1997)?"), use the clarification to narrow down the list and return
        a smaller list of candidates (hopefully just 1!)

        - If the clarification uniquely identifies one of the movies, this should return a 1-element
        list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it should return a list
        with the indices it could be referring to (to continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by the clarification

        @ Julia
        """
        newCandidates = []
        clarification = clarification.lower()

        for i in candidates:
            if clarification.lower() in self.titles[i][0].lower():
                newCandidates.append(i) 

        """
        for i in candidates:
            inputs = clarification.split()
            for c in inputs:
                if re.search(c, self.titles[i][0].lower(), flags = re.IGNORECASE): 
                    newCandidates.append(i)
                elif re.search(c, self.titles[i][1].lower(), flags = re.IGNORECASE):
                    newCandidates.append(i)
        """
        return newCandidates

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use any
        attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered positive

        :returns: a binarized version of the movie-rating matrix

        @Max
        """
        #############################################################################
        # TODO: Binarize the supplied ratings matrix. Do not use the self.ratings   #
        # matrix directly in this function.                                         #
        #############################################################################

        # The starter code returns a new matrix shaped like ratings but full of zeros.
        binarized_ratings = np.where(ratings > 2.5, 1, -1)
        binarized_ratings[np.where(ratings == 0)] = 0


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors

        @Julia
        @Kayla Ella
        """
        #############################################################################
        # TODO: Compute cosine similarity between the two vectors.
        #############################################################################
        numerator = np.dot(u,v)
        denominator = np.linalg.norm(u) * np.linalg.norm(v)
        if denominator == 0:
            return 0
        similarity = numerator/denominator
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return similarity


    def remove_articles(self, text):
        articles = [" a ", " an ", " the "]
        for a in articles:
            text = text.replace(a, " ")
        caps = ["A ", "The ", "An "]
        for a in caps:
            text = text.replace(a, "")

        return text
    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in ratings_matrix,
          in descending order of recommendation

        @Ella @Max @Julia
        """

        #######################################################################################
        # TODO: Implement a recommendation function that takes a vector user_ratings          #
        # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
        # Do not use the self.ratings matrix directly in this function.                       #
        # For starter mode, you should use item-item collaborative filtering                  #
        # with cosine similarity, no mean-centering, and no normalization of scores.          #
        #######################################################################################

        # Populate this list with k movie indices to recommend to the user.
        recommendations = [0] * k
        rated = []
        predicted = {}

        for m in range(len(user_ratings)):
            if user_ratings[m] != 0:
                rated.append(m)

        for i in range(len(user_ratings)):
            rxi = 0.0
            if i not in rated:
                for j in range(len(user_ratings)):
                    rxj = user_ratings[j]
                    sij = self.similarity(ratings_matrix[j], ratings_matrix[i])
                    rxi += rxj * sij
                predicted[i] = rxi
        top = sorted(((value, key) for (key, value) in predicted.items()), reverse = True)
        for i in range (k):
            recommendations[i] = top[i][1]

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return recommendations

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
        """Return debug information as a string for the line string from the REPL"""
        # Pass the debug information that you may think is important for your
        # evaluators
        debug_info = disambiguate ("1997", [1359, 2716])
        return debug_info

    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    # @Kayla
    #############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your chatbot
        can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6 instructions.
        Remember: in thee starter mode, movie names will come in quotation marks and
        expressions of sentimnt will be simple!
        Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, run:')
    print('    python3 repl.py')
