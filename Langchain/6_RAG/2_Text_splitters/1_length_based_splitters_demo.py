from langchain_text_splitters import CharacterTextSplitter


text = """
One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.

Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.

It's obviously true that the returns for performance are superlinear in business. Some think this is a flaw of capitalism, and that if we changed the rules it would stop being true. But superlinear returns for performance are a feature of the world, not an artifact of rules we've invented. We see the same pattern in fame, power, military victories, knowledge, and even benefit to humanity. In all of these, the rich get richer. [1]

You can't understand the world without understanding the concept of superlinear returns. And if you're ambitious you definitely should, because this will be the wave you surf on.

It may seem as if there are a lot of different situations with superlinear returns, but as far as I can tell they reduce to two fundamental causes: exponential growth and thresholds.

The most obvious case of superlinear returns is when you're working on something that grows exponentially. For example, growing bacterial cultures. When they grow at all, they grow exponentially. But they're tricky to grow. Which means the difference in outcome between someone who's adept at it and someone who's not is very great.

Startups can also grow exponentially, and we see the same pattern there. Some manage to achieve high growth rates. Most don't. And as a result you get qualitatively different outcomes: the companies with high growth rates tend to become immensely valuable, while the ones with lower growth rates may not even survive.

Y Combinator encourages founders to focus on growth rate rather than absolute numbers. It prevents them from being discouraged early on, when the absolute numbers are still low. It also helps them decide what to focus on: you can use growth rate as a compass to tell you how to evolve the company. But the main advantage is that by focusing on growth rate you tend to get something that grows exponentially.

YC doesn't explicitly tell founders that with growth rate "you get out what you put in," but it's not far from the truth. And if growth rate were proportional to performance, then the reward for performance p over time t would be proportional to pt.

Even after decades of thinking about this, I find that sentence startling.
"""

# define splitter object
splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0, separator="")


result = splitter.split_text(text)
print(result)
"""Output:
['One of th', 'e most imp', 'ortant thi', 'ngs I didn', "'t underst", 'and about', 'the world', 'when I was', 'a child i', 's the degr', 'ee to whic', 'h the retu', 'rns for pe', 'rformance', 'are superl', 'inear.\n\nTe', 'achers and', 'coaches i', 'mplicitly', 'told us th', 'e returns', 'were linea', 'r. "You ge', 't out," I', 'heard a th', 'ousand tim', 'es, "what', 'you put in', '." They me', 'ant well,', 'but this i', 's rarely t', 'rue. If yo', 'ur product', 'is only h', 'alf as goo', 'd as your', 'competitor', "'s, you do", "n't get ha", 'lf as many', 'customers', '. You get', 'no custome', 'rs, and yo', 'u go out o', 'f business', ".\n\nIt's ob", 'viously tr', 'ue that th', 'e returns', 'for perfor', 'mance are', 'superlinea', 'r in busin', 'ess. Some', 'think this', 'is a flaw', 'of capita', 'lism, and', 'that if we', 'changed t', 'he rules i', 't would st', 'op being t', 'rue. But s', 'uperlinear', 'returns f', 'or perform', 'ance are a', 'feature o', 'f the worl', 'd, not an', 'artifact o', 'f rules we', "'ve invent", 'ed. We see', 'the same', 'pattern in', 'fame, pow', 'er, milita', 'ry victori', 'es, knowle', 'dge, and e', 'ven benefi', 't to human', 'ity. In al', 'l of these', ', the rich', 'get riche', 'r. [1]\n\nYo', "u can't un", 'derstand t', 'he world w', 'ithout und', 'erstanding', 'the conce', 'pt of supe', 'rlinear re', 'turns. And', "if you're", 'ambitious', 'you defin', 'itely shou', 'ld, becaus', 'e this wil', 'l be the w', 'ave you su', 'rf on.\n\nIt', 'may seem', 'as if ther', 'e are a lo', 't of diffe', 'rent situa', 'tions with', 'superline', 'ar returns', ', but as f', 'ar as I ca', 'n tell the', 'y reduce t', 'o two fund', 'amental ca', 'uses: expo', 'nential gr', 'owth and t', 'hresholds.', 'The most', 'obvious c', 'ase of sup', 'erlinear r', 'eturns is', "when you'r", 'e working', 'on somethi', 'ng that gr', 'ows expone', 'ntially. F', 'or example', ', growing', 'bacterial', 'cultures.', 'When they', 'grow at al', 'l, they gr', 'ow exponen', 'tially. Bu', "t they're", 'tricky to', 'grow. Whic', 'h means th', 'e differen', 'ce in outc', 'ome betwee', 'n someone', "who's adep", 't at it an', 'd someone', "who's not", 'is very gr', 'eat.\n\nStar', 'tups can a', 'lso grow e', 'xponential', 'ly, and we', 'see the s', 'ame patter', 'n there. S', 'ome manage', 'to achiev', 'e high gro', 'wth rates.', "Most don'", 't. And as', 'a result y', 'ou get qua', 'litatively', 'different', 'outcomes:', 'the compa', 'nies with', 'high growt', 'h rates te', 'nd to beco', 'me immense', 'ly valuabl', 'e, while t', 'he ones wi', 'th lower g', 'rowth rate', 's may not', 'even survi', 've.\n\nY Com', 'binator en', 'courages f', 'ounders to', 'focus on', 'growth rat', 'e rather t', 'han absolu', 'te numbers', '. It preve', 'nts them f', 'rom being', 'discourage', 'd early on', ', when the', 'absolute', 'numbers ar', 'e still lo', 'w. It also', 'helps the', 'm decide w', 'hat to foc', 'us on: you', 'can use g', 'rowth rate', 'as a comp', 'ass to tel', 'l you how', 'to evolve', 'the compan', 'y. But the', 'main adva', 'ntage is t', 'hat by foc', 'using on g', 'rowth rate', 'you tend', 'to get som', 'ething tha', 't grows ex', 'ponentiall', 'y.\n\nYC doe', "sn't expli", 'citly tell', 'founders', 'that with', 'growth rat', 'e "you get', 'out what', 'you put in', '," but it\'', 's not far', 'from the t', 'ruth. And', 'if growth', 'rate were', 'proportion', 'al to perf', 'ormance, t', 'hen the re', 'ward for p', 'erformance', 'p over ti', 'me t would', 'be propor', 'tional to', 'pt.\n\nEven', 'after deca', 'des of thi', 'nking abou', 't this, I', 'find that', 'sentence s', 'tartling.']"""
print(type(result))  # <class 'list'>
