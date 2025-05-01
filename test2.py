docs = [
    "in this video i would like to start the discussion about convolutional new networks",
    "which is another architecture of neural networks that we are going to see specifically kindof engineered to address problems that we are facing in computer vision",
    "i want to start this discussion with just showing you a picture and if i ask you to tell me what would actually be the first object that you pay attention to then most people will probably respond to with yellow cab and its not really accident that lot of cabs inb and its not really accident that lot of cabs in a lotofcapitals are painted yellow u",
    "and of course that attention to bright colors originates from thousands of years of evolution where people are of course trained to pay attention to bright colors which are typically sources of food such as fruits or in some instances sources of danger such like the yellow tiger that is coming towards us so kind of joking aside",
    "i think you can associate this kind image with the following kind of processing of various",
    "with the following kind of processing of various kindof stage processing that is been going on in your brain trying to from the kind of perception system that embeds into it that we have embedded into it",
    "so if you can imagine the process then the first thing that you open your eyes soon that you have your eyes closed and you see an image like this then the brain is actually in a first few milliseconds",
    "is sampling this image in a course kind of way and determines whether we have situations as an imdetermines whether we have situations as an immediate kind of threat or a source of food and then very quickly the brain switches softly into objects paying attention to objects which are associated with the task that we have",
    "we we have to execute so we have the for example if youre waiting for someone in this kindof scene then you start paying more attention to people coming towards you people getting out of vehicles and so on",
    "so now what i like to start getting into is the mechanics of  a litke to start getting into is the mechanics of a little bit of computer vision some kind of basic principles and i wanted",
    "to cover a little bit you know the question as to okay what is an image that like the one weve seen earlier and how were going to represent it and evidently an image is  a matrix",
    "and i think we had some discussion about images before and if a black white imag is or — – a grayscale image as you actually see over here is  the ma which includes integer numbers and this integerhose elements are integer numbers and this integer numbers typically we are",
    "you know each element corresponds to a pixel and the dynamic range that we associate with typically for in computer vision with those pixels is 8 bits so we represent the information at each pixel and codes as 8 bits which means that these integer numbers are zero from 0",
    "to 255 and so when we go to color im ese we need more than one of those matrices in fact we need three matricels",
    "typically again there are sensorspically again there are sensors which are of course encode the information to far more than three matrices",
    "but those three matricels are corresponds to the fundamental colors typically of red green and blue and this which is actually what you see here",
    "and also probably you notice that these numbers are now floating point numbers",
    "lets say from 0 to one and this results from normalizing these pixels numbers to with a number 255 because 2 to the power of 8 is 256 and therefore all our numbers will be from 06 and therefore all our numbers will be from 0 to 255 in those matrices",
    "if we start dividing every element with 255 we get numbers between 0 and 1",
    "and this is basically what we needto do in order of process the images in with with our new neural networks that we will introduce now called the convolutional neural networks",
    "so with convolutional neural networks we need to start the discussion on what is a convolution and this is whats coming next"
]

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP

# Use a pre-trained model from sentence-transformers
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models as well, like 'paraphrase-MiniLM-L6-v2'

embeddings = embedding_model.encode(docs)
print(embeddings)
print(embedding_model.similarity(embeddings, embeddings))