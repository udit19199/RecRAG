Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 11 items.

=== ITEM id=9ceb0cc80761e918 ===
Gold sentences and facts:
Sentence 0: Montague (Monty) Banks (15 July 1897 [registered on 18 July 1897] – 7 January 1950 born Mario Bianchi) was an Italian comedian and film director who achieved success in the United States and in England.
  Fact 0.0: Montague (Monty) Banks was born Mario Bianchi.
  Fact 0.1: Montague (Monty) Banks was born 15 July 1897.
  Fact 0.2: Montague (Monty) Banks' birth was registered on 18 July 1897.
  Fact 0.3: Montague (Monty) Banks died on 7 January 1950.
  Fact 0.4: Montague (Monty) Banks was an Italian comedian.
  Fact 0.5: Montague (Monty) Banks was an Italian film director.
  Fact 0.6: Montague (Monty) Banks achieved success in the United States.
  Fact 0.7: Montague (Monty) Banks achieved success in England.
Sentence 1: George Seaton (April 17, 1911 – July 28, 1979) was an American screenwriter, playwright, film director and producer, and theatre director.
  Fact 1.0: George Seaton was born April 17, 1911.
  Fact 1.1: George Seaton died July 28, 1979.
  Fact 1.2: George Seaton was an American screenwriter.
  Fact 1.3: George Seaton was an American playwright.
  Fact 1.4: George Seaton was an American film director.
  Fact 1.5: George Seaton was an American film producer.
  Fact 1.6: George Seaton was an American theatre director.

Retrieved items:
<item 1>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 1>

<item 2>
The Black Hand Gang
The Black Hand Gang is a 1930 British comedy film directed by Monty Banks and starring Wee Georgie Wood, Viola Compton and Alfred Wood.

 It was made by British International Pictures and based on a play by "Black Hand George" by Bert Lee and R.P. Weston.
</item 2>

<item 3>
Ken Barrie
Leslie Hulme (9 January 1933 – 29 July 2016), known professionally as Ken Barrie, was an English voice actor and singer best known for narrating, and singing
the theme tune of the BBC television programmes "Postman Pat" and "Charlie Chalk".

 He was also responsible for providing the voices of several of the series' characters.
</item 3>

<item 4>
Mark Benton
Mark Benton (born 16 November 1965) is an English actor, known for his roles as Eddie in "Early Doors", Howard in "Northern Lights" and Martin Pond in
"Barbara".

 Benton has also starred in BBC One drama, "Waterloo Road" as maths teacher, Daniel Chalk from 2011 to 2013 and again in 2014.
</item 4>

<item 5>
Keep Your Seats, Please
Keep Your Seats, Please is a 1936 British comedy film directed by Monty Banks and starring George Formby, Florence Desmond and Alastair Sim.

 It marked the film debut of the child star Binkie Stuart.

 The film was made by Associated Talking Pictures.
</item 5>

<item 6>
The Church Mouse
The Church Mouse is a 1934 British comedy film directed by Monty Banks and starring Laura La Plante, Ian Hunter and Edward Chapman.

 It was made by the British subsidiary of Warner Brothers at the company's Teddington Studios.

 It was made as a more expensive production than much of the studio's low-budget quota quickie output.
</item 6>

<item 7>
George Seaton
George Seaton (April 17, 1911 – July 28, 1979) was an American screenwriter, playwright, film director and producer, and theatre director.
</item 7>

<item 8>
Joey Bishop
Joseph Abraham Gottlieb (February 3, 1918 – October 17, 2007), known professionally as Joey Bishop, was an American entertainer who appeared on television as
early as 1948 and eventually starred in his own weekly comedy series playing a talk show host, then later hosted a late night talk show.

 He later became a member of the "Rat Pack" with Frank Sinatra, Peter Lawford, Sammy Davis, Jr., and Dean Martin.
</item 8>

<item 9>
Monty Banks
Montague (Monty) Banks (15 July 1897 [registered on 18 July 1897] – 7 January 1950 born Mario Bianchi) was an Italian comedian and film director who achieved
success in the United States and in England.
</item 9>

<item 10>
Everett Greenbaum
Everett Greenbaum (December 20, 1919 – July 11, 1999) was an American television and film writer and actor who contributed to such shows as "The Andy Griffith
Show" (24 Episodes), "M*A*S*H" (35 Episodes), "Love American Style", "The Real McCoys" (32 Episodes), "Sanford and Son", and "The George Gobel Show".

 Greenbaum was a co-creator with Jim Fritzell of "Mister Peepers" an important early television show which starred Wally Cox.

 He wrote the Hollywood feature film "Good Neighbor Sam", as well as a series of films starring Don Knotts that included "The Shakiest Gun in the West", "The
Reluctant Astronaut", and "The Ghost and Mr. Chicken".
</item 10>
=== END ITEM id=9ceb0cc80761e918 ===

=== ITEM id=2b7958307b09c29a ===
Gold sentences and facts:
Sentence 0:  In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.
  Fact 0.0: The Eleanor and John Y. Brown Jr. Foundation gave a grant in 1972.
  Fact 0.1: The 1972 grant from the Eleanor and John Y. Brown Jr. Foundation led to the construction of the current facility.
Sentence 1:  He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.
  Fact 1.0: The 55th governor of Kentucky served from 1979 to 1983.
  Fact 1.1: The 55th governor of Kentucky may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.

Retrieved items:
<item 1>
Rochelle Buffenstein
Rochelle (Shelley) Buffenstein is a staff scientist at Calico, an Alphabet, Inc. funded research outfit investigating aging.

 Previously, she had been a professor of Physiology at theBarshop Institute for Longevity and Aging Studies at the University of Texas Health Science Center at
San Antonio.

 Her research focuses on comparative vertebrate physiology, energetics, cancer biology, and aging.

 She has worked with marsupials, mole-rats, tenrecs, bats, subterranean mammals, and primates.

 Her best known work involves exceptional aging -- specifically, why naked mole-rats live for so much longer than other rodents.

 Her lab has investigated theories of aging including oxidative damage theory, the advanced glycation end product theory, and the telomere theory.
</item 1>

<item 2>
Rubin Schron
Rubin Schron, who goes by Ruby, is a New York City real estate investor, landlord and the founder of Cammeby's International Group.

 He has a personal net worth of well over $10 billion, according to data company Real Capital Analytics.

 The portfolio of Cammeby's, which Schron founded in 1967, includes office buildings, market-rate and government-subsidized apartment complexes, nursing homes,
the 16-building complex in Sunset Park now known as Industry City, a stake in the bottom half of Woolworth Building and industrial properties scattered across
Long Island.

 In 2013, Schron made an unsolicited and unsuccessful offer to buy the Empire State Building for $2 billion, but he has not had problems closing many other
deals throughout his long career.

 In 2003, an investment group led by Schron paid $600 million for a portfolio of about 6,000 outer-borough apartments from Donald Trump.
</item 2>

<item 3>
Panera Bread
Panera Bread Company is an American chain of bakery-café fast casual restaurants in the United States and Canada.

 Its headquarters are in Sunset Hills, Missouri, a suburb of St. Louis, and operates as Saint Louis Bread Company in the St. Louis metropolitan area.

 Offerings include soups, salads, pasta, sandwiches, specialty drinks, and bakery items.
</item 3>

<item 4>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 4>

<item 5>
Sanders–Brown Center on Aging
The Sanders–Brown Center on Aging at the University of Kentucky in Lexington, Kentucky, is responsible for research, education and service programs in aging.

 The program was founded in 1963 with the creation of the Council on Aging.

 In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.

 The center was named for the Browns and his then-business partner, (Col.) Harlan Sanders.

 The four-story brick and concrete structure fronts South Limestone at the edge of the medical campus.

 The complex features 32 research laboratories that employs more than 150.
</item 5>

<item 6>
John Y. Brown Jr.
John Young Brown Jr. (born December 28, 1933) is an American politician, entrepreneur, and businessman from the U.S. state of Kentucky.

 He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a
multimillion-dollar restaurant chain.
</item 6>

<item 7>
McAlister's Deli
McAlister's Deli is an American chain of fast casual restaurants founded in 1989 in Oxford, Mississippi by retired dentist Dr. Don Newcomb.

 There are currently over 400 locations in 26 states, ranging from Virginia in the East to Florida in the South to Arizona in West to Michigan in the North.

 The menu includes deli sandwiches, "Texas-size" spuds (baked potatoes), soups, salads, and desserts, as well as catering items such as sandwich trays and boxed
lunches.

 The chain is also known for its McAlister's Famous Sweet Tea, which is available by the glass or by the gallon.
</item 7>

<item 8>
VIP's
VIP's, alternatively written Vip's, is a defunct restaurant chain in the Western United States that operated from 1968 until the late 1980s, based in Salem,
Oregon.

 With more than 50 locations, it was once the largest restaurant chain based in Oregon.

 It was a Denny's-style restaurant, a type that was commonly known at that time as a "coffee shop" but is now more commonly known as a casual dining restaurant.

 Most restaurants were located near freeways and were open 24 hours.

 At its peak, the chain had locations in five states: Oregon, Washington, Idaho, Nevada and northern California.
</item 8>

<item 9>
Benjamin A. Rogge
Rogge served for many years as a Liberty Fund trustee.

 Thomas Sowell gives Rogge credit for encouraging him to write a book on economics and race.

 Rogge also was a frequent presenter at the seminars of the Foundation for Economic Education (FEE).

 FEE's founder, Leonard Read, thought of Rogge as Read's eventual successor, an outcome prevented by Read outliving Rogge.

 An extended (but somewhat grainy) video clip of a Rogge FEE lecture on "Competition and Monopoly" on YouTube illustrates the dry wit that made him a popular
speaker.

 Rogge attended 13 meetings of the influential international Mont Pelerin Society.

 Rogge helped produce, and narrated, a documentary on Adam Smith that was funded by Liberty Fund.

 Rogge wrote the introduction to a collection of quotations from Adam Smith.

 A collection of Rogge's speeches, often on topics in economics or education, was published under the title "Can Capitalism Survive?"

 Wabash College, where he taught for many years, established a speaker series in his honor.
</item 9>

<item 10>
Benjamin A. Rogge
Benjamin A. Rogge (June 18, 1920 – November 17, 1980) was an American economist, college administrator, and libertarian writer, speaker and foundation advisor.

 Rogge received an A.B. degree from Hastings College and an M.A. from the University of Nebraska Lincoln.

 Rogge received his PhD in economics from Northwestern.

 At Wabash College, Rogge taught in the summer Institute for Professional Development, in addition to his usual teaching in economics.

 Rogge co-authored an economics principles textbook with John Van Sickle.

 One strength of the text is the account that it gives of Joseph Schumpeter's process of creative destruction.

 Rogge helped organize a series of lectures by Milton Friedman at Wabash that were eventually developed into Friedman's "Capitalism and Freedom" book.

 Much later, Rogge participated in a brainstorming session for Friedman's Free to Choose television series.

 Liberty Fund was founded with money from Pierre Goodrich, who sought advice from Rogge during the Fund's early years.

 Rogge served for many years as a Liberty Fund trustee.
</item 10>
=== END ITEM id=2b7958307b09c29a ===

=== ITEM id=5cc329cf697d47fc ===
Gold sentences and facts:
Sentence 0: Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris Institute or Art School, Harris Technical School and the Harris Orphanage.
  Fact 0.0: Edmund Robert Harris lived c. 1804 to 27 May 1877.
  Fact 0.1: Edmund Robert Harris was an English lawyer.
  Fact 0.2: Edmund Robert Harris was from Preston, Lancashire, UK.
  Fact 0.3: Edmund Robert Harris was the principal benefactor of the Harris Museum.
  Fact 0.4: Edmund Robert Harris was the principal benefactor of the Harris Institute or Art School.
  Fact 0.5: Edmund Robert Harris was the principal benefactor of the Harris Technical School.
  Fact 0.6: Edmund Robert Harris was the principal benefactor of the Harris Orphanage.
Sentence 1: The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
  Fact 1.0: The Harris Museum, Art Gallery & Preston Free Public Library is a museum building in Preston.
  Fact 1.1: The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed building.

Retrieved items:
<item 1>
Rice Northwest Museum of Rocks and Minerals
The Rice Northwest Museum of Rocks and Minerals is a non-profit museum in Hillsboro, Oregon, United States.

 Located just north of the Sunset Highway on the northern edge of Hillsboro, the earth science museum is in the Portland metropolitan area.

 Opened in 1997, the museum’s collections date to the 1930s with the museum housed in a home built to display the rock and mineral collections of the museum
founders.

 The ranch style home is listed on the National Register of Historic Places, the first of its kind listed in Oregon.

 In 2015 the museum became a Smithsonian Affiliate museum.
</item 1>

<item 2>
Harris Museum
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
</item 2>

<item 3>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 3>

<item 4>
Edmund Robert Harris
Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris
Institute or Art School, Harris Technical School and the Harris Orphanage.
</item 4>

<item 5>
Edmund de Unger
Edmund Robert Anthony de Unger (Hungarian: "Odon Antal Robert de Unger" , b 6 August 1918, Budapest - d 25 January 2011, Ham, Surrey) was a Hungarian-born
property developer and art collector.

 In London he built up the Keir Collection, one of the greatest post-war collections of Islamic art, bequeathed in 2008 to the Pergamon Museum of Islamic Art in
Berlin.

 The arrangement for the museum to curate the collection came to an end in July 2012.

 The collection is now hosted by the Dallas Museum of Art as of May 2014 for a 15-year renewable loan.
</item 5>

<item 6>
Hanbys Corner, Delaware
The last Hanby to occupy the property, Albert T. Hanby (1881-1947), another son of Samuel, attended West Chester State College before getting his law degree
from Penn Law School.

 Albert became a Philadelphia lawyer and left his farm at Hanby's Corner to be used for the good of "all the children in Delaware".

 He and his wife created a foundation in 1945 to protect the property from further development.

 Today the YMCA operates their Hanby Camp there, and the trust provide scholarships for kids who might not otherwise be able to attend.

 In 2007 that support was over $70,000.
</item 6>

<item 7>
Museum of Jurassic Technology
The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.

 It was founded by David Hildebrand Wilson and Diana Drake Wilson (husband and wife) in 1988.
</item 7>

<item 8>
J. Robert Donnelly Husky Heritage Sports Museum
The J. Robert Donnelly Husky Heritage Sports Museum is a public museum located on the University of Connecticut's main campus at Storrs, Connecticut.

 The museum documents and celebrates UConn's intercollegiate athletics.

 Opening its doors on January 19, 2002, the 2,700-square-foot museum was named in honor of benefactor and 1940 Connecticut basketball and football captain J.
Robert (Bob) Donnelly (1971-2005).

 Exhibits include national and regional championship trophies, trading cards, photographs, and various sports memorabilia, as well as a six-screen video wall
replaying moments of triumph for the UConn Huskies.

 A life-size fiberglass sculpture of Jonathan the Husky, UConn's mascot, greets visitors at the museum entrance.

 The Connecticut Basketball Rotunda, featuring NCAA championship trophies and life-size cutouts of Ray Allen and Rebecca Lobo, is among the museum's permanent
exhibits.
</item 8>

<item 9>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 9>

<item 10>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 10>
=== END ITEM id=5cc329cf697d47fc ===

=== ITEM id=3860912022338754 ===
Gold sentences and facts:
Sentence 0: Ravi Sethi (born 1947) is an Indian computer scientist retired from Bell Labs and president of Avaya Labs Research.
  Fact 0.0: Ravi Sethi was born in 1947.
  Fact 0.1: Ravi Sethi is an Indian computer scientist.
  Fact 0.2: Ravi Sethi retired from Bell Labs.
  Fact 0.3: Ravi Sethi is president of Avaya Labs Research.
Sentence 1: Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company, owned by Finnish company Nokia.
  Fact 1.0: Nokia Bell Labs was formerly named AT&T Bell Laboratories.
  Fact 1.1: Nokia Bell Labs was formerly named Bell Telephone Laboratories.
  Fact 1.2: Nokia Bell Labs was formerly named Bell Labs.
  Fact 1.3: Nokia Bell Labs is an American research and scientific development company.
  Fact 1.4: Nokia Bell Labs is owned by the Finnish company Nokia.
Sentence 2:  Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
  Fact 2.0: The headquarters of Nokia Bell Labs are located in Murray Hill, New Jersey.
  Fact 2.1: Nokia Bell Labs has other laboratories around the rest of the United States.
  Fact 2.2: Nokia Bell Labs has laboratories in other countries.

Retrieved items:
<item 1>
Genpact
Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.

 It’s listed on the New York Stock Exchange under the symbol G.In 2016, the company reported net revenues of US$2.57 billion with more than 77,000 employees in
20 countries.

 Genpact provides consulting services to hundreds of clients, including approximately one-fifth of the Fortune Global 500.
</item 1>

<item 2>
Umesh Varshney
Umesh Varshney (born 1957) is an Indian molecular biologist, academician and the head of the "Prof.Umesh Varshney's Lab" at the Indian Institute of Science,
Bengaluru.

 He is a J. C. Bose National Fellow of the Department of Science and Technology and is known for his studies on protein synthesis and DNA repair in "Escherichia
coli" and Mycobacterium tuberculosis.

 An elected fellow of the Indian Academy of Sciences, Indian National Science Academy and the National Academy of Sciences (India), he is also a recipient of
the National Bioscience Award for Career Development of the Government of India.

 The Council of Scientific and Industrial Research, the apex agency of the Government of India for scientific research, awarded him the Shanti Swarup Bhatnagar
Prize for Science and Technology, one of the highest Indian science awards, in 2001, and then in 2014 with the G. N. Ramachandran Gold Medal for Excellence in
Biological Sciences & Technology for his contributions to biological sciences.
</item 2>

<item 3>
Ravi Sethi
Ravi Sethi (born 1947) is an Indian computer scientist retired from Bell Labs and president of Avaya Labs Research.

 He also serves as a member of the National Science Foundation's Computer and Information Science and Engineering (CISE) Advisory Committee.

 He is best known as one of three authors of the classic computer science textbook "", also known as the "Dragon Book".
</item 3>

<item 4>
Richard Yoo
Richard Yoo is an American entrepreneur, the co-founder and former CEO of the web hosting company Rackspace, and the founder and former CEO of web hosting
company ServerBeach.
</item 4>

<item 5>
Pharma Medica
Pharma Medica Research Inc. is an American research and development company that works with pharmaceutical companies and performs clinical trials in the
pharmaceutical and biotechnology industry.

 The company is based in Toronto, Canada with headquarters in Mississauga, Ontario.

 Pharma Medica has locations in Scarborough, Ontario and St. Charles, Missouri.
</item 5>

<item 6>
Bell Labs
Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company,
owned by Finnish company Nokia.

 Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
</item 6>

<item 7>
Sethi–Ullman algorithm
In computer science, the Sethi–Ullman algorithm is an algorithm named after Ravi Sethi and Jeffrey D. Ullman, its inventors, for translating abstract syntax
trees into machine code that uses as few registers as possible.
</item 7>

<item 8>
Von Braun Center for Science &amp; Innovation
The Von Braun Center for Science & Innovation (VCSI ) is a non-profit research and development organization based in Huntsville, Alabama and is named for
pioneering aerospace engineer Wernher von Braun.

 VCSI is affiliated with NASA, Department of Defense and other federal government agencies.
</item 8>

<item 9>
Center for Climate Systems Research (CCSR)
The Center for Climate Systems Research is a key Earth Institute center with over 25 scientists and staff researching issues involving the interplay between
Earth's climate and society.

 The center's motto is "Columbia's Gateway to NASA and Beyond", as it has a special relationship with the NASA Goddard Institute for Space Studies (GISS).

 It is a "Gateway to NASA", because the center is co-located with GISS in Columbia University's Armstrong Hall above Tom's Restaurant in New York City.

 In this historic location, Columbia and NASA scientists work together closely to understand the Earth's climate and how changes can impact society both in the
United States and around the world.

 Importantly, CCSR scientists also work with many other partners – governmental, private sector, and non-governmental organizations – to answer these
challenging scientific and socioeconomic questions (i.e., "and Beyond").

 The director of CCSR is Dr. Michael J. Puma.
</item 9>

<item 10>
David H. Bailey
David Harold Bailey (born 1948) is a mathematician and computer scientist.

 He received his B.S. in mathematics from Brigham Young University in 1972 and his Ph.D. in mathematics from Stanford University in 1976.

 He worked for 14 years as a computer scientist at NASA Ames Research Center, but since 1998 has been at the Lawrence Berkeley National Laboratory.

 He is now officially retired, but continues as an active researcher.

 He is also a Research Associate at the University of California, Davis, Department of Computer Science.
</item 10>
=== END ITEM id=3860912022338754 ===

=== ITEM id=db4373cf78f7d41e ===
Gold sentences and facts:
Sentence 0: Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
  Fact 0.0: The Spanish name of Teide National Park is 'Parque nacional del Teide'.
  Fact 0.1: Teide National Park is a national park.
  Fact 0.2: Teide National Park is located in Tenerife.
  Fact 0.3: Tenerife is in the Canary Islands, Spain.
Sentence 1: Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands (Spain).
  Fact 1.0: The Spanish name of Garajonay National Park is 'Parque nacional de Garajonay'.
  Fact 1.1: Garajonay National Park is located in the center of the island of La Gomera.
  Fact 1.2: Garajonay National Park is located in the north of the island of La Gomera.
  Fact 1.3: La Gomera is one of the Canary Islands in Spain.

Retrieved items:
<item 1>
Komodo National Park
Komodo National Park is a national park in Indonesia located within the Lesser Sunda Islands in the border region between the provinces of East Nusa Tenggara
and West Nusa Tenggara.

 The park includes the three larger islands Komodo, Padar and Rinca, and 26 smaller ones, with a total area of 1,733 km (603 km of it land).

 The national park was founded in 1980 to protect the Komodo dragon, the world's largest lizard.

 Later it was dedicated to protecting other species, including marine species.

 In 1991 the national park was declared a UNESCO World Heritage Site.
</item 1>

<item 2>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 2>

<item 3>
Guadarrama National Park
Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the
fifth largest in Spain's national parks system.

 The Guadarrama mountain range ("Sistema Central") contains some ecologically valuable areas, located in the Community of Madrid and Castile and León (provinces
of Segovia and Ávila).

 The law that regulates the recently approved national park was published in the BOE in  26, 2013 (2013--) .
</item 3>

<item 4>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 4>

<item 5>
Badiar National Park
Badiar National Park (French: "Parc National du Badiar" ) is a national park in Guinea, on the border with Senegal and contiguous with Senegal's much larger
Niokolo-Koba National Park.

 It was established on 30 May 1985 (by ordonnance N°124/PRG/85), partly in response to Senegal's concern about poaching in Niokolo-Koba National Park.

 Badiar is an International Union for Conservation of Nature Category II park.
</item 5>

<item 6>
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 6>

<item 7>
North Cascades National Park
North Cascades National Park is a U.S. National Park located in the state of Washington.

 The park is the largest of the three National Park Service units that comprise the North Cascades National Park Service Complex.

 Several national wilderness areas and British Columbia parkland adjoin the National Park.

 The park features rugged mountain peaks and protects portions of the North Cascades range.
</item 7>

<item 8>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 8>

<item 9>
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
</item 9>

<item 10>
Landes de Gascogne Regional Natural Park
Landes de Gascogne Regional Natural Park (French: "Parc naturel régional des Landes de Gascogne") is a protected area of pine forest, wetland and oceanic
coastline located in the Aquitaine region of southwestern France.
</item 10>
=== END ITEM id=db4373cf78f7d41e ===

=== ITEM id=fd7af6a3f33896e5 ===
Gold sentences and facts:
Sentence 0:  The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.
  Fact 0.0: The group included Alexander Bogdanov.
  Fact 0.1: The group included Maxim Gorky.
  Fact 0.2: The group included Anatoly Lunacharsky.
  Fact 0.3: The group included Mikhail Pokrovsky.
  Fact 0.4: The group included Grigory Aleksinsky.
  Fact 0.5: The group included Stanislav Volski.
  Fact 0.6: The group included Martyn Liadov.
Sentence 1: Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.
  Fact 1.0: Anatoly Vasilyevich Lunacharsky's Russian name is Анато́лий Васи́льевич Лунача́рский.
  Fact 1.1: Anatoly Vasilyevich Lunacharsky was born on 23 November 1875 (Old Style 11 November 1875).
  Fact 1.2: Anatoly Vasilyevich Lunacharsky died on 26 December 1933.
  Fact 1.3: Anatoly Vasilyevich Lunacharsky was a Russian Marxist revolutionary.
  Fact 1.4: Anatoly Vasilyevich Lunacharsky was the first Soviet People's Commissar of Education.
  Fact 1.5: Anatoly Vasilyevich Lunacharsky was responsible for culture and education as People's Commissar of Education.

Retrieved items:
<item 1>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 1>

<item 2>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 2>

<item 3>
Graph query row:
person: Anatoly Lunacharsky
relationship: MEMBER_OF
connected_to: Vpered
birth_date: 23 November [O.S. 11 November] 1875
death_date: 26 December 1933
birthplace: None
</item 3>

<item 4>
Matvey Skobelev
Matvey Ivanovich Skobelev (Russian: Матве́й Ива́нович Ско́белев ; November 9, 1885, Baku – July 29, 1938, Moscow) was a Russian Marxist revolutionary and
politician.
</item 4>

<item 5>
Martemyan Ryutin
Martemyan Nikitich Ryutin (Russian: Мартемья́н Ники́тич Рю́тин ) (1890–1937) was a Russian Marxist revolutionary, and a political functionary of the Russian
Communist Party.

 Ryutin is best remembered as the leader of a pro-peasant political faction organized against Soviet leader Joseph Stalin in the early 1930s and as the primary
author of a 200-page oppositional platform.

 Ryutin was arrested by the Soviet secret police, along with his co-thinkers, in what has come to be known as the Ryutin Affair.

 He was executed in January 1937 as part of the Yezhovshchina (Great Purge) conducted against political oppositionists and suspected economic "wreckers" and
spies.
</item 5>

<item 6>
Nikolay Speshnev
Nikolay Alexandrovich Speshnev (Russian: Николай Александрович Спе́шнев; 1821, Kursk - 1882, St. Petersburg) was a 19th-century Russian aristocrat and political
activist, best known for his involvement with the pro-socialist literary discussion group the Petrashevsky Circle.

 He formed a secret revolutionary society from among the members of the circle, which included the young Fyodor Dostoevsky.

 After the government of Tsar Nicholas I arrested the members of the Petrashevsky Circle in 1849, Speshnev was interrogated, threatened with torture, and
eventually sentenced, along with Dostoevsky, Petrashevsky and others, to execution by firing squad.

 The sentence was commuted to hard labour in Siberia, but the prisoners were only informed of this after enduring a mock execution.
</item 6>

<item 7>
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 7>

<item 8>
Vladimir Bazarov
Vladimir Alexandrovich Bazarov ( Влади́мир Алекса́ндрович База́ров; 1874–1939) was a Russian Marxist revolutionary, journalist, philosopher, and economist, born
Vladimir Alexandrovich Rudnev.

 Bazarov is best remembered as a pioneer in the development of economic planning in the Soviet Union.
</item 8>

<item 9>
Vpered
Vpered ("Forward" or "Hasten") (1909 - 1912) was an organization emanating from within the Russian Social Democratic Labour Party (Russian Social Democracy or
RSDLP).

 The faction was gathered by Alexander Bogdanov (1873 - 1928) in December 1909.

 The group included: Alexander Bogdanov, Maxim Gorky, Anatoly Lunacharsky, Mikhail Pokrovsky, Grigory Aleksinsky, Stanislav Volski, and Martyn Liadov.

 Although Vpered emerged from the Bolshevik wing of Russian Social Democracy, the group was critical of Lenin.
</item 9>

<item 10>
Petrashevsky Circle
The Petrashevsky Circle was a Russian literary discussion group of progressive-minded commoner-intellectuals in St. Petersburg in the 1840s.

 It was organized by Mikhail Petrashevsky, a follower of the French utopian socialist Charles Fourier.

 Among the members were writers, teachers, students, minor government officials, and army officers.

 While differing in political views, most of them were opponents of the tsarist autocracy and Russian serfdom.

 Like that of the Lyubomudry group founded earlier in the century, the purpose of the circle was to discuss Western philosophy and literature that was
officially banned by the Imperial government of Tsar Nicholas I.
</item 10>
=== END ITEM id=fd7af6a3f33896e5 ===

=== ITEM id=d5cf87d13bc58b2b ===
Gold sentences and facts:
Sentence 0: John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.
  Fact 0.0: John Alexander was born on 5 October 1955.
  Fact 0.1: John Alexander is English.
  Fact 0.2: John Alexander is a former footballer.
  Fact 0.3: John Alexander holds the post of club secretary at Manchester United.
Sentence 1:  Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in 1976.
  Fact 1.0: John Alexander was born in Liverpool.
  Fact 1.1: John Alexander began his football career with a club called Ulysses.
  Fact 1.2: John Alexander was later picked up by Millwall.
  Fact 1.3: John Alexander made his league debut for Millwall in 1976.
Sentence 2:  He is the uncle of Liverpool player Trent Alexander-Arnold.
  Fact 2.0: John Alexander is the uncle of Liverpool player Trent Alexander-Arnold.
Sentence 3: Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.
  Fact 3.0: Trent John Alexander-Arnold was born on 7 October 1998.
  Fact 3.1: Trent John Alexander-Arnold is English.
  Fact 3.2: Trent John Alexander-Arnold is a professional footballer.
  Fact 3.3: Trent John Alexander-Arnold plays as a right-back.
  Fact 3.4: Trent John Alexander-Arnold plays for Premier League club Liverpool.

Retrieved items:
<item 1>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 1>

<item 2>
William McCourty
William McCourty (1884 – 10 December 1917) was an English professional footballer born in Morpeth, Northumberland, who played in the Football League for
Birmingham.

 McCourty, a former coal miner who played at left half, joined Birmingham in May 1909.

 He made his debut in the Second Division on 13 September 1909, deputising for the injured Tommy Daykin in a home game against Glossop which finished as a 2–2
draw, and was never picked again, returning to non-league football the following year.
</item 2>

<item 3>
Nick Powell
Nicholas Edward "Nick" Powell (born 23 March 1994) is an English professional footballer who plays for Wigan Athletic as a midfielder.

 He began his career at Crewe Alexandra, initially as a forward, making his debut at the age of 16.

 After impressing for Crewe Alexandra during the 2011–12 season, including scoring in their 2–0 victory in the League Two play-off Final, he moved to Manchester
United in July 2012.

 Unable to break into the Manchester United first team, however, he went on loan to Wigan Athletic, Leicester City and Hull City before being released in June
2016.

 He joined Wigan Athletic in July 2016.

 He has represented England at under-16, under-17, under-18, under-19 and under-21 levels.
</item 3>

<item 4>
John Logan (footballer, born 1912)
John William Logan (16 August 1912 – October 1980) was an English professional footballer born in Horden, near Peterlee, County Durham who played in the
Football League for Darlington, Barnsley and Sheffield Wednesday.

 He played as a wing half.
</item 4>

<item 5>
Jack Charlton
John Charlton, OBE, DL (born 8 May 1935) is an English former footballer and manager who played as a defender.

 He was part of the England team that won the 1966 World Cup.

 He is the elder brother of former Manchester United forward Bobby Charlton, who was also a teammate in England's World Cup final victory.

 He spent his entire club career with Leeds United from 1950 to 1973, helping the club to the Second Division title (1963–64), First Division title (1968–69),
FA Cup (1972), League Cup (1968), Charity Shield (1969), Inter-Cities Fairs Cup (1968 and 1971), as well as one other promotion from the Second Division
(1955–56) and five second-place finishes in the First Division, two FA Cup final defeats and one Inter-Cities Fairs Cup final defeat.

 His 629 league and 762 total competitive appearances are club records.

 In 2006, Leeds United supporters voted Charlton into the club's greatest ever XI.
</item 5>

<item 6>
Trent Alexander-Arnold
Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.

 He is an academy graduate of Liverpool and made his senior debut for the club in 2016, aged 18.

 The following year, he won the club's Young Player of the Season award.

 He has also represented England at all youth levels from under-16 through to under-19.

 Due to the length of his name, his surname is shortened to Alexander for Premier League fixtures.
</item 6>

<item 7>
Charlie Leatherbarrow
Charles Leatherbarrow (born Letherbarrow) (16 January 1870 – 26 December 1940) was an English professional footballer born in Banbury who played in the Football
League for Rotherham Town, Walsall Town Swifts and Small Heath.

 He played at inside right or centre forward.
</item 7>

<item 8>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 8>

<item 9>
John Alexander (footballer, born 1955)
John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.

 Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in
1976.

 After scoring twice in 15 appearances over the course of two years with Millwall, he moved to Reading; he enjoyed greater success with Reading, scoring nine
goals in 25 games.

 In 1981, he joined Northampton Town, but spent just one season with them before retiring from football at the age of 26 in 1982.

 Upon retiring from football, Alexander got a job with the BBC, but he later returned to football as club secretary at Watford.

 In 2000, he took up the same post at Tottenham Hotspur, spending 10 years there before applying for the same job at Manchester United, where he would replace
the retiring Ken Ramsden.

 He took over at Manchester United on 1 July 2010.

 He is the uncle of Liverpool player Trent Alexander-Arnold.
</item 9>

<item 10>
Tommy Bell (footballer, born 1906)
Thomas Bell (9 November 1906 – 1983) was an English professional footballer born in Seaham Harbour.

 He could play at inside right, centre forward and right half and was a regular goalscorer throughout his professional career.
</item 10>
=== END ITEM id=d5cf87d13bc58b2b ===

=== ITEM id=40680578890c1d17 ===
Gold sentences and facts:
Sentence 0:  Her personal best throw is 58.87 metres, achieved at the 2002 Asian Games in Busan.
  Fact 0.0: Her personal best throw is 58.87 metres.
  Fact 0.1: She achieved her personal best throw at the 2002 Asian Games.
  Fact 0.2: The 2002 Asian Games were held in Busan.
Sentence 1:  A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.
  Fact 1.0: A total of 460 athletes took part in the competition.
  Fact 1.1: The athletes comprised 281 men and 179 women.
  Fact 1.2: The athletes came from 39 nations.

Retrieved items:
<item 1>
Caster Semenya
Mokgadi Caster Semenya Bronze OIB (born 7 January 1991) is a South African middle-distance runner and 2016 Olympic gold medallist.

 Semenya won gold in the women's 800 metres at the 2009 World Championships with a time of 1:55.45 and at the 2017 World Championships in her new personal best,
1:55.16.

 Semenya also won silver medals at the 2011 World Championships and the 2012 Summer Olympics, both in the 800 metres.

 She was the winner of the gold medal in the 800 metre event at the 2016 Summer Olympics.
</item 1>

<item 2>
Zhao Qinggang
Zhao Qinggang (; born 24 July 1985) is a Chinese track and field athlete who competes in the javelin throw.

 His 89.15 m personal best throw, set in 2014, is the Chinese record.

 It was also the Asian record until surpassed in 2017.

 He represented his country at the 2013 World Championships in Athletics.

 He is the 2014 Asian Games champion.

 At the East Asian Games he was the runner-up in 2009 and winner in 2013.

 He is a two-time Chinese champion (2012 and 2013) and was the winner of the 2013 National Games of China.
</item 2>

<item 3>
Liu Ailing
Liu Ailing (; born June 2, 1967) is a Chinese former footballer who played for the China national team at the 1991, 1995 and 1999 editions of the FIFA Women's
World Cup.

 She won a silver medal at the 1996 Atlanta Olympics and participated at the 2000 Sydney Olympics.

 A playmaking midfielder, she played professional club football in Japan and the United States.
</item 3>

<item 4>
Masayoshi Yamamoto
Masayoshi Yamamoto (Japanese: 山本 雅賢) is a Japanese male artistic gymnast, representing his nation at international competitions.

 He won the gold medal in the Pommel horse event at the 2014 Asian Games in Incheon, South Korea.
</item 4>

<item 5>
Elaine Schreiber
Elaine Annette Schreiber (4 June 1939 – 11 June 2017) was an Australian Paralympic table tennis player and athlete.

 She contracted poliomyelitis as a child.

 At the 1964 Tokyo Games, she won a gold medal in the women's club throw A event, a silver medal in the women's javelin A event, and a bronze medal in the
women's singles B event.

 At the 1968 Tel Aviv Games, she won a silver medal in the women's doubles C event with Marion O'Brien; she also competed but did not win any medals in athletic
events at the 1968 games.

 She participated without winning medals in athletics events at both the 1972 Heidelberg and 1976 Toronto Games, and also participated in table tennis at the
latter games.
</item 5>

<item 6>
Lee Young-sun
Lee Young-Sun (born 21 February 1974) is a retired female javelin thrower from South Korea.

 Her personal best throw is 58.87 metres, achieved at the 2002 Asian Games in Busan.
</item 6>

<item 7>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 7>

<item 8>
Luan Zhili
Luan Zhili (born 6 January 1973) is a female discus thrower from PR China.

 She finished ninth at the 1997 World Championships and won the 1998 Asian Games, the latter in a personal best throw of 63.43 metres.

 The Chinese, and Asian, record is currently held by Xiao Yanling with 71.68 metres.
</item 8>

<item 9>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 9>

<item 10>
John Lindsay (Paralympian)
At the 2004 Athens Games, he came seventh in the first round of the Men's 100 m T53 event and sixth in the third round of the Men's 200 m T53 event.

 He was an Australian Institute of Sport scholarship holder in 1995 and 2000.
</item 10>
=== END ITEM id=40680578890c1d17 ===

=== ITEM id=363831e96314cf42 ===
Gold sentences and facts:
Sentence 0:  It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black matipo", "Pittosporum tenuifolium", is not related to "Myrsine").
  Fact 0.0: Myrsine contains about 200 species.
  Fact 0.1: Myrsine includes several notable radiations.
  Fact 0.2: The matipo of New Zealand is one of the notable radiations within Myrsine.
  Fact 0.3: The kōlea of Hawaiʻi is one of the notable radiations within Myrsine.
  Fact 0.4: The New Zealand 'black matipo', Pittosporum tenuifolium, is not related to Myrsine.
Sentence 1: Storax or snowbell is the common names of Styrax, a genus of about 130 species of large shrubs or small trees in the family Styracaceae, mostly native to warm temperate to tropical regions of the Northern Hemisphere, with the majority in eastern and southeastern Asia, but also crossing the equator in South America.
  Fact 1.0: Storax or snowbell are the common names of Styrax.
  Fact 1.1: Styrax is a genus of about 130 species.
  Fact 1.2: Styrax species are large shrubs or small trees.
  Fact 1.3: Styrax belongs to the family Styracaceae.
  Fact 1.4: Styrax is mostly native to warm temperate to tropical regions of the Northern Hemisphere.
  Fact 1.5: The majority of Styrax species are in eastern and southeastern Asia.
  Fact 1.6: Styrax also crosses the equator in South America.

Retrieved items:
<item 1>
Pterostyrax hispidus
Pterostyrax hispidus, the epaulette tree or fragrant epaulette tree, is a species of flowering plant in the family Styracaceae, native to China and Japan.

 Growing to 15 m tall by 12 m broad, it is a substantial, spreading, deciduous shrub with oval leaves up to 20 cm long, and clusters of pure white, fragrant,
pendent flowers in summer.
</item 1>

<item 2>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 2>

<item 3>
Styrax fraserensis
Styrax fraserensis is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is a tree endemic to Peninsular Malaysia.

 It is threatened by habitat loss.
</item 3>

<item 4>
Rapanea
Rapanea is a genus of plant in family Primulaceae.

 It has often been placed in synonymy with "Myrsine", and many species have been moved to "Myrsine".
</item 4>

<item 5>
Styrax vilcabambae
Styrax vilcabambae (syn.

 "Pamphilia vilcabambae" D.R.Simpson) is a species of flowering plant in the genus "Styrax" and the family Styracaceae.

 It is endemic to Peru.
</item 5>

<item 6>
Styrax
Storax or snowbell is the common names of Styrax, a genus of about 130 species of large shrubs or small trees in the family Styracaceae, mostly native to warm
temperate to tropical regions of the Northern Hemisphere, with the majority in eastern and southeastern Asia, but also crossing the equator in South America.

 The resin obtained from the tree is called storax or benzoin (not to be confused with the "Liquidambar" storax balsam).
</item 6>

<item 7>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 7>

<item 8>
Aizoaceae
Aizoaceae (the fig-marigold family) is a large family of dicotyledonous flowering plants containing 135 genera and about 1900 species.

 They are commonly known as stone plants or carpet weeds.

 They are often called vygies in South Africa and New Zealand.

 Species that resemble stones or pebbles are sometimes called mesembs.

 Several species are known as ice plants because of the glistening globular bladder cells covering their stems, fruit and leaves, "... they sparkle like ice
crystals."
</item 8>

<item 9>
Styrax litseoides
Styrax litseoides is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is endemic to Vietnam.
</item 9>

<item 10>
Styrax crotonoides
Styrax crotonoides is a species of plant in the genus "Styrax" and family Styracaceae.

 It is found in Peninsular Malaysia and Singapore.
</item 10>
=== END ITEM id=363831e96314cf42 ===

=== ITEM id=46824e8bedd844fa ===
Gold sentences and facts:
Sentence 0:  In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.
  Fact 0.0: The Eleanor and John Y. Brown Jr. Foundation gave a grant in 1972.
  Fact 0.1: The 1972 grant from the Eleanor and John Y. Brown Jr. Foundation led to the construction of the current facility.
Sentence 1:  He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.
  Fact 1.0: The 55th governor of Kentucky served from 1979 to 1983.
  Fact 1.1: The 55th governor of Kentucky may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.

Retrieved items:
<item 1>
McAlister's Deli
McAlister's Deli is an American chain of fast casual restaurants founded in 1989 in Oxford, Mississippi by retired dentist Dr. Don Newcomb.

 There are currently over 400 locations in 26 states, ranging from Virginia in the East to Florida in the South to Arizona in West to Michigan in the North.

 The menu includes deli sandwiches, "Texas-size" spuds (baked potatoes), soups, salads, and desserts, as well as catering items such as sandwich trays and boxed
lunches.

 The chain is also known for its McAlister's Famous Sweet Tea, which is available by the glass or by the gallon.
</item 1>

<item 2>
VIP's
VIP's, alternatively written Vip's, is a defunct restaurant chain in the Western United States that operated from 1968 until the late 1980s, based in Salem,
Oregon.

 With more than 50 locations, it was once the largest restaurant chain based in Oregon.

 It was a Denny's-style restaurant, a type that was commonly known at that time as a "coffee shop" but is now more commonly known as a casual dining restaurant.

 Most restaurants were located near freeways and were open 24 hours.

 At its peak, the chain had locations in five states: Oregon, Washington, Idaho, Nevada and northern California.
</item 2>

<item 3>
Panera Bread
Panera Bread Company is an American chain of bakery-café fast casual restaurants in the United States and Canada.

 Its headquarters are in Sunset Hills, Missouri, a suburb of St. Louis, and operates as Saint Louis Bread Company in the St. Louis metropolitan area.

 Offerings include soups, salads, pasta, sandwiches, specialty drinks, and bakery items.
</item 3>

<item 4>
Sanders–Brown Center on Aging
The Sanders–Brown Center on Aging at the University of Kentucky in Lexington, Kentucky, is responsible for research, education and service programs in aging.

 The program was founded in 1963 with the creation of the Council on Aging.

 In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.

 The center was named for the Browns and his then-business partner, (Col.) Harlan Sanders.

 The four-story brick and concrete structure fronts South Limestone at the edge of the medical campus.

 The complex features 32 research laboratories that employs more than 150.
</item 4>

<item 5>
Rubin Schron
Rubin Schron, who goes by Ruby, is a New York City real estate investor, landlord and the founder of Cammeby's International Group.

 He has a personal net worth of well over $10 billion, according to data company Real Capital Analytics.

 The portfolio of Cammeby's, which Schron founded in 1967, includes office buildings, market-rate and government-subsidized apartment complexes, nursing homes,
the 16-building complex in Sunset Park now known as Industry City, a stake in the bottom half of Woolworth Building and industrial properties scattered across
Long Island.

 In 2013, Schron made an unsolicited and unsuccessful offer to buy the Empire State Building for $2 billion, but he has not had problems closing many other
deals throughout his long career.

 In 2003, an investment group led by Schron paid $600 million for a portfolio of about 6,000 outer-borough apartments from Donald Trump.
</item 5>

<item 6>
John Y. Brown Jr.
John Young Brown Jr. (born December 28, 1933) is an American politician, entrepreneur, and businessman from the U.S. state of Kentucky.

 He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a
multimillion-dollar restaurant chain.
</item 6>

<item 7>
Aladdin's Eatery
Aladdin's Eatery is a chain of franchised restaurants in the American Midwest and South-East, specializing in Lebanese cuisine.

 Adapted to American tastes, the sites are fast casual restaurants that also offer take out.
</item 7>

<item 8>
Benjamin A. Rogge
Benjamin A. Rogge (June 18, 1920 – November 17, 1980) was an American economist, college administrator, and libertarian writer, speaker and foundation advisor.

 Rogge received an A.B. degree from Hastings College and an M.A. from the University of Nebraska Lincoln.

 Rogge received his PhD in economics from Northwestern.

 At Wabash College, Rogge taught in the summer Institute for Professional Development, in addition to his usual teaching in economics.

 Rogge co-authored an economics principles textbook with John Van Sickle.

 One strength of the text is the account that it gives of Joseph Schumpeter's process of creative destruction.

 Rogge helped organize a series of lectures by Milton Friedman at Wabash that were eventually developed into Friedman's "Capitalism and Freedom" book.

 Much later, Rogge participated in a brainstorming session for Friedman's Free to Choose television series.

 Liberty Fund was founded with money from Pierre Goodrich, who sought advice from Rogge during the Fund's early years.

 Rogge served for many years as a Liberty Fund trustee.
</item 8>

<item 9>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 9>

<item 10>
Rochelle Buffenstein
Rochelle (Shelley) Buffenstein is a staff scientist at Calico, an Alphabet, Inc. funded research outfit investigating aging.

 Previously, she had been a professor of Physiology at theBarshop Institute for Longevity and Aging Studies at the University of Texas Health Science Center at
San Antonio.

 Her research focuses on comparative vertebrate physiology, energetics, cancer biology, and aging.

 She has worked with marsupials, mole-rats, tenrecs, bats, subterranean mammals, and primates.

 Her best known work involves exceptional aging -- specifically, why naked mole-rats live for so much longer than other rodents.

 Her lab has investigated theories of aging including oxidative damage theory, the advanced glycation end product theory, and the telomere theory.
</item 10>
=== END ITEM id=46824e8bedd844fa ===

=== ITEM id=e357c4777e6d7def ===
Gold sentences and facts:
Sentence 0: Darby O'Gill and the Little People is a 1959 Walt Disney Productions feature film starring Albert Sharpe, Janet Munro, Sean Connery and Jimmy O'Dea, in a tale about a wily Irishman and his battle of wits with leprechauns.
  Fact 0.0: Darby O'Gill and the Little People is a feature film.
  Fact 0.1: Darby O'Gill and the Little People was made by Walt Disney Productions.
  Fact 0.2: Darby O'Gill and the Little People was released in 1959.
  Fact 0.3: Darby O'Gill and the Little People stars Albert Sharpe.
  Fact 0.4: Darby O'Gill and the Little People stars Janet Munro.
  Fact 0.5: Darby O'Gill and the Little People stars Sean Connery.
  Fact 0.6: Darby O'Gill and the Little People stars Jimmy O'Dea.
  Fact 0.7: Darby O'Gill and the Little People tells a tale about a wily Irishman.
  Fact 0.8: The Irishman in Darby O'Gill and the Little People has a battle of wits with leprechauns.
Sentence 1:  Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.
  Fact 1.0: The film named in this sentence was produced by Walt Disney Pictures.
  Fact 1.1: The film named in this sentence was produced by Aamir Khan Productions.
  Fact 1.2: The film named in this sentence was produced by UTV Motion Pictures.
  Fact 1.3: Aamir Khan stars as the father in the film named in this sentence.

Retrieved items:
<item 1>
Frankie Darro
Frankie Darro (born Frank Johnson, Jr.

 December 22, 1917 – December 25, 1976) was an American actor and later in his career a stuntman.

 He began his career as a child actor in silent films, progressed to lead roles and co-starring roles in adventure, western, dramatic, and comedy films, and
later became a character actor and voice-over artist.

 He is known for his role as Lampwick, the unlucky boy turned into a donkey in Walt Disney's 2nd animated feature, "Pinocchio", which was originally released in
February 1940.
</item 1>

<item 2>
Darby O'Gill and the Little People
Darby O'Gill and the Little People is a 1959 Walt Disney Productions feature film starring Albert Sharpe, Janet Munro, Sean Connery and Jimmy O'Dea, in a tale
about a wily Irishman and his battle of wits with leprechauns.

 The film was directed by Robert Stevenson and its screenplay written by Lawrence Edward Watkin after the books of Herminie Templeton Kavanagh.
</item 2>

<item 3>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 3>

<item 4>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 4>

<item 5>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 5>

<item 6>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 6>

<item 7>
Pinocchio (1940 film)
Pinocchio is a 1940 American animated musical fantasy film produced by Walt Disney Productions and based on the Italian children's novel "The Adventures of
Pinocchio" by Carlo Collodi.

 It was the second animated feature film produced by Disney, made after the success of "Snow White and the Seven Dwarfs" (1937).
</item 7>

<item 8>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
</item 8>

<item 9>
Walt Disney's Treasury of Classic Tales
Walt Disney's Treasury of Classic Tales is an American Sunday comic strip, which ran in newspapers from 13 July 1952 until 15 February 1987.

 Each story adapted a different Disney film, such as Darby O'Gill and the Little People, Peter Pan, or Davy Crockett.

 It was run in relatively few papers with 58 in 1957 and 55 in 1966 and was principally a vehicle for promoting new and re-released Disney films.
</item 9>

<item 10>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 10>
=== END ITEM id=e357c4777e6d7def ===
