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

=== ITEM id=0840ee380584f497 ===
Gold sentences and facts:
Sentence 0:  He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems."
  Fact 0.0: The recipient was awarded the 2012 Nobel Prize in Physics.
  Fact 0.1: The recipient shared the 2012 Nobel Prize in Physics jointly with Serge Haroche.
  Fact 0.2: The 2012 Nobel Prize in Physics was awarded for ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems.
Sentence 1:  Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.
  Fact 1.0: Since 2001, Serge Haroche has been a Professor at the Collège de France.
  Fact 1.1: Since 2001, Serge Haroche has held the Chair of Quantum Physics.

Retrieved items:
<item 1>
François Englert
He is the recipient of the 2013 Prince of Asturias Award in technical and scientific research, together with Peter Higgs and the CERN.
</item 1>

<item 2>
The entries profile Danish popular-science author Tor Nørretranders, who introduced “exformation,” and British science journalist David Bradley. They describe
popular science, the Thomas Merton justice award, Swedish scholar and feminist Ebba Witt-Brattström, children’s series Barna Hedenhös, the Camanachd Cup’s
Albert Smith Medal, illustrator Jenny Nyström, and the historic French journal *Annales de chimie et de physique*. The remaining entries outline Nobel
</item 2>

<item 3>
Brian Josephson
Brian David Josephson, {'1': ", '2': ", '3': ", '4': "} (born 4 January 1940), is a Welsh theoretical physicist and professor emeritus of physics at the
University of Cambridge.

 Best known for his pioneering work on superconductivity and quantum tunnelling, he was awarded the Nobel Prize in Physics in 1973 for his prediction of the
Josephson effect, made in 1962 when he was a 22-year-old PhD student at Cambridge University.

 Josephson is the only Welshman to have won a Nobel Prize in Physics.

 He shared the prize with physicists Leo Esaki and Ivar Giaever, who jointly received half the award for their own work on quantum tunnelling.
</item 3>

<item 4>
List of Nobel laureates in Chemistry
The Nobel Prize in Chemistry (Swedish: "Nobelpriset i kemi" ) is awarded annually by the Royal Swedish Academy of Sciences to scientists in the various fields
of chemistry.

 It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel, who died in 1896.

 These prizes are awarded for outstanding contributions in chemistry, physics, literature, peace, and physiology or medicine.

 As dictated by Nobel's will, the award is administered by the Nobel Foundation and awarded by a committee that consists of five members elected by the Royal
Swedish Academy of Sciences.

 The first Nobel Prize in Chemistry was awarded in 1901 to Jacobus Henricus van 't Hoff, of the Netherlands.

 Each recipient receives a medal, a diploma and a monetary award prize that has varied throughout the years.

 In 1901, van 't Hoff received 150,782 SEK, which is equal to 7,731,004 SEK in December 2007.

 The award is presented in Stockholm at an annual ceremony on December 10, the anniversary of Nobel's death.
</item 4>

<item 5>
Nobel Prize in Physics
The Nobel Prize in Physics (Swedish: "Nobelpriset i fysik" ) is a yearly award given by the Royal Swedish Academy of Sciences for those who conferred the most
outstanding contributions for mankind in the field of physics.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel
Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 5>

<item 6>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 6>

<item 7>
Ekhard Salje: Cambridge mineralogist and former Earth Sciences head. Peter Doherty: Australian veterinary immunologist; 1995 Lasker, 1996 medicine Nobel with
Zinkernagel, 1997 Australian of the Year. Antje Boetius: German geomicrobiologist; 2009 Leibniz laureate, studies seabed methane-oxidizing microbes and climate.
Brian Josephson: Welsh physicist; 1973 physics Nobel for the Joseph
</item 7>

<item 8>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 8>

<item 9>
Serge Haroche
Serge Haroche (born 11 September 1944) is a French physicist who was awarded the 2012 Nobel Prize for Physics jointly with David J. Wineland for
"ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems", a study of the particle of light, the photon.

 This and his other works developed laser spectroscopy.

 Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.

 In 1971 he defended his doctoral thesis in physics at the University of Paris VI, his research has been conducted under the direction of Claude
Cohen-Tannoudji.
</item 9>

<item 10>
David J. Wineland
David Jeffrey Wineland (born February 24, 1944) is an American Nobel-laureate physicist at the National Institute of Standards and Technology (NIST) physics
laboratory.

 His work has included advances in optics, specifically laser cooling trapped ions and using ions for quantum computing operations.

 He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of
individual quantum systems."
</item 10>
=== END ITEM id=0840ee380584f497 ===

=== ITEM id=61e3c759a62d4e9e ===
Gold sentences and facts:
Sentence 0:  In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.
  Fact 0.0: The Eleanor and John Y. Brown Jr. Foundation gave a grant in 1972.
  Fact 0.1: The 1972 grant from the Eleanor and John Y. Brown Jr. Foundation led to the construction of the current facility.
Sentence 1:  He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.
  Fact 1.0: The 55th governor of Kentucky served from 1979 to 1983.
  Fact 1.1: The 55th governor of Kentucky may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.

Retrieved items:
<item 1>
Panera Bread
Panera Bread Company is an American chain of bakery-café fast casual restaurants in the United States and Canada.

 Its headquarters are in Sunset Hills, Missouri, a suburb of St. Louis, and operates as Saint Louis Bread Company in the St. Louis metropolitan area.

 Offerings include soups, salads, pasta, sandwiches, specialty drinks, and bakery items.
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
The material surveys fast-casual, casual-dining, and fast-food chains. Chipotle, listed as CMG, serves tacos and burritos internationally; McAlister’s has 400+
U.S. delis; Singapore-founded 4FINGERS sells crispy chicken across four countries; Captain D’s specializes in seafood; and Aladdin’s Eatery serves Lebanese
food. Other examples include global fried-chicken giant KFC, India’s Nirula’s, historic Chicken in the Rough,
</item 3>

<item 4>
Henry Sawyer was a Philadelphia lawyer, civil-rights activist, and councilman whose Supreme Court arguments shaped Establishment Clause law. Kellogg executive
A. D. David Mackay was CEO from 2006 to 2011. George McKelvey and Dannel McCollum were Ohio and Illinois mayors; McCollum also wrote about his mother’s landmark
case. William King won Maine’s first gubernatorial election unopposed in 1820. Other entries cover Kentucky’s Sanders–Brown aging center
</item 4>

<item 5>
Panera Bread is a U.S.–Canada bakery-café chain headquartered near St. Louis, where it is called Saint Louis Bread Company. Indian entrepreneur Prem Ganapathy
founded Dosa Plaza, growing it to 45 outlets across four countries. Quebec’s independently franchised La Belle Province is sometimes open 24 hours. Gino’s
Hamburgers began in Baltimore in 1957 and relaunched in 2010. West Virginia-based Tudor’s Biscuit World serves breakfast and homestyle
</item 5>

<item 6>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 6>

<item 7>
Joseph Saunders led Visa; Gertrude Vanderbilt Whitney was a sculptor and founded the Whitney Museum. Roscoe Channing was a Princeton All-American, Rough Rider,
and mining executive. Fayez Sarofim is a Houston billionaire investor and Texans co-owner; Charles Agemian was a senior banker. The Rockefeller Brothers Fund,
founded in 1940, supports philanthropy. Bankers William Harrison and Walter Shipley led major JPMorgan predecessor mergers. Rockefeller family members include
philanthropists
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
Sanders–Brown Center on Aging
The Sanders–Brown Center on Aging at the University of Kentucky in Lexington, Kentucky, is responsible for research, education and service programs in aging.

 The program was founded in 1963 with the creation of the Council on Aging.

 In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.

 The center was named for the Browns and his then-business partner, (Col.) Harlan Sanders.

 The four-story brick and concrete structure fronts South Limestone at the edge of the medical campus.

 The complex features 32 research laboratories that employs more than 150.
</item 9>

<item 10>
McAlister's Deli
McAlister's Deli is an American chain of fast casual restaurants founded in 1989 in Oxford, Mississippi by retired dentist Dr. Don Newcomb.

 There are currently over 400 locations in 26 states, ranging from Virginia in the East to Florida in the South to Arizona in West to Michigan in the North.

 The menu includes deli sandwiches, "Texas-size" spuds (baked potatoes), soups, salads, and desserts, as well as catering items such as sandwich trays and boxed
lunches.

 The chain is also known for its McAlister's Famous Sweet Tea, which is available by the glass or by the gallon.
</item 10>
=== END ITEM id=61e3c759a62d4e9e ===

=== ITEM id=7a39b8aebb1bec53 ===
Gold sentences and facts:
Sentence 0: Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.
  Fact 0.0: Tender Fictions is a 1996 autobiographical documentary film.
  Fact 0.1: Tender Fictions was directed by American experimental filmmaker Barbara Hammer.
Sentence 1: Bougainville – Our Island Our Fight is a 1998 Australian documentary film.
  Fact 1.0: Bougainville – Our Island Our Fight is a 1998 Australian documentary film.
Sentence 2:  It was produced and directed by Wayne Coles-Janess.
  Fact 2.0: Bougainville – Our Island Our Fight was produced by Wayne Coles-Janess.
  Fact 2.1: Bougainville – Our Island Our Fight was directed by Wayne Coles-Janess.

Retrieved items:
<item 1>
Life at the End of the Rainbow
Life at the End of the Rainbow is a 2002 documentary by Australian filmmaker Wayne Coles-Janess about the small farming community of Rainbow, population 500,
which lies on the edge of the Big Desert, North Western Victoria, Australia.

 It is 55 minutes long.
</item 1>

<item 2>
Wayne Coles-Janess
Wayne Coles-Janess is an Australian producer, writer and director of drama and documentary film and TV programs.

 Based in Melbourne, Australia, he has produced documentaries about frontier places in the country.

 He has also made some documentaries in several international locations, including during times of war.
</item 2>

<item 3>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 3>

<item 4>
Bougainville – Our Island Our Fight
Bougainville – Our Island Our Fight is a 1998 Australian documentary film.

 It was produced and directed by Wayne Coles-Janess.
</item 4>

<item 5>
Waiting for Happiness
The film presents typical Mauritanian moments of beauty, struggle, alienation, and humor, which are experienced by groups socially divided from each other, such
as Bidhan women drinking tea and gossiping, West African migrants passing through Mauritania to get to Europe (and finding an unsuccessful comrade washed
ashore).

 The young protagonist who has returned interacts with all of these groups as an outsider, as he struggles to remember even his own Hassaniya Arabic dialect,
but prefers instead French.

 Many of the themes and characters presage Sissako's 2014 film Timbuktu, and both explore liminal Sahel identities authentically situated in everyday life.

 Waiting for Happiness premiered at the 2002 Cannes Film Festival in the Un Certain Regard section.
</item 5>

<item 6>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 6>

<item 7>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 7>

<item 8>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 8>

<item 9>
The Coconut Revolution
The Coconut Revolution is a 2001 multi-award winning documentary film about the struggle of the indigenous peoples of Bougainville Island during the
Bougainville Civil War.

 The movement is described as the "world's first successful eco-revolution" and has drawn parallels with the conflict depicted in the 2009 film, "Avatar".
</item 9>

<item 10>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 10>
=== END ITEM id=7a39b8aebb1bec53 ===

=== ITEM id=f647dceea04f558f ===
Gold sentences and facts:
Sentence 0: Steve Rash is an American film director and producer best known for directing such films as "The Buddy Holly Story", "Can't Buy Me Love", "Queens Logic", "" and "".
  Fact 0.0: Steve Rash is an American film director.
  Fact 0.1: Steve Rash is an American producer.
  Fact 0.2: Steve Rash is best known for directing The Buddy Holly Story.
  Fact 0.3: Steve Rash is best known for directing Can't Buy Me Love.
  Fact 0.4: Steve Rash is best known for directing Queens Logic.
  Fact 0.5: Steve Rash is best known for directing two further films whose titles are blank in the source text.
Sentence 1: Can't Buy Me Love is a 1987 American teen romantic comedy feature film starring Patrick Dempsey and Amanda Peterson in a story about a nerd at a high school in Tucson, Arizona who gives a cheerleader $1,000 to pretend to be his girlfriend for a month.
  Fact 1.0: Can't Buy Me Love is a 1987 American film.
  Fact 1.1: Can't Buy Me Love is a teen romantic comedy feature film.
  Fact 1.2: Can't Buy Me Love stars Patrick Dempsey.
  Fact 1.3: Can't Buy Me Love stars Amanda Peterson.
  Fact 1.4: The story of Can't Buy Me Love is about a nerd at a high school in Tucson, Arizona.
  Fact 1.5: In Can't Buy Me Love, the nerd gives a cheerleader $1,000 to pretend to be his girlfriend for a month.

Retrieved items:
<item 1>
Can't Buy Me Love (film)
Can't Buy Me Love is a 1987 American teen romantic comedy feature film starring Patrick Dempsey and Amanda Peterson in a story about a nerd at a high school in
Tucson, Arizona who gives a cheerleader $1,000 to pretend to be his girlfriend for a month.

 The film was directed by Steve Rash and takes its title from a Beatles song with the same title.
</item 1>

<item 2>
Steve Rash
Steve Rash is an American film director and producer best known for directing such films as "The Buddy Holly Story", "Can't Buy Me Love", "Queens Logic", "" and
"".
</item 2>

<item 3>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 3>

<item 4>
Bring It On: In It to Win It
Bring It On: In It to Win It is a 2007 teen film directed by Steve Rash and starring Ashley Benson, Cassie Scerbo and Michael Copon.
</item 4>

<item 5>
Josh Charles
Joshua Aaron "Josh" Charles (born September 15, 1971) is an American stage, film and television actor.

 He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his
early work as Knox Overstreet in "Dead Poets Society".
</item 5>

<item 6>
Robert Oliveri
Robert Dane Oliveri (born April 28, 1978) is a former American actor.

 He is best known for his portrayal of Nick Szalinski in the 1989 Disney film, "Honey, I Shrunk the Kids", for which he was nominated for a Young Artist Award
and a Saturn Award.

 He reprised the role in the 1992 sequel, "Honey, I Blew Up the Kid" and the 3D film, "Honey, I Shrunk the Audience".

 He is also known as Kevin, Winona Ryder's little brother, in Tim Burton's "Edward Scissorhands".
</item 6>

<item 7>
Howard Rosenman
Howard Rosenman (born February 1, 1945), also known as Zvi Howard Rosenman, is an American producer and motion picture executive.

 He specializes in producing romantic comedy films and documentary films.

 Some of his most popular productions include "Father of the Bride" (1991) starring Steve Martin and Diane Keaton, Joss Whedon's "Buffy the Vampire Slayer"
(1992) and "The Family Man" (2000) starring Nicolas Cage.

 Rosenman's documentary film "" won the Peabody Award and the 1990 Academy Award for Best Documentary Feature; his film "The Celluloid Closet" also won the
Peabody Award.
</item 7>

<item 8>
Jimmy Tatro
James Richard Tatro (born February 16, 1992) is an American actor, comedian, writer and YouTube personality.

 He is the creator and star of the YouTube channel "LifeAccordingToJimmy", which has about 2.7 million subscribers and over 440 million video views.

 Tatro writes, produces and directs each of his video sketches with his friend, Christian A. Pierce.

 Tatro has appeared in "Divergent", "Grown Ups 2", "22 Jump Street" and "", and will appear in the film "Camp Manna" alongside Gary Busey, set to be released in
2017.

 He is also starring in the true-crime satire, "American Vandal", in which he plays Dylan Maxwell, the accused vandal.

 He will also appear in the movie, Action No. 1, a film about a group of people who attempt to steal the first comic book with Superman off the actor Nicholas
Cage.
</item 8>

<item 9>
Just Friends
Just Friends is a 2005 American romantic comedy film directed by Roger Kumble, written by Adam 'Tex' Davis and starring Ryan Reynolds, Amy Smart, Anna Faris,
Chris Klein and Christopher Marquette.

 The plot focuses on a formerly overweight high school nerd (Reynolds) who attempts to free himself from the friend zone after reconnecting with his lifelong
crush and best friend (Smart) while visiting his hometown for Christmas.
</item 9>

<item 10>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 10>
=== END ITEM id=f647dceea04f558f ===

=== ITEM id=c825a0091d50e6e7 ===
Gold sentences and facts:
Sentence 0: Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live artist and former international management consultant.
  Fact 0.0: Eenasul Fateh was born on 3 April 1959.
  Fact 0.1: Eenasul Fateh's Bengali name is ঈনাসুল ফাতেহ.
  Fact 0.2: Eenasul Fateh is also known by his stage name Aladin.
  Fact 0.3: Eenasul Fateh is a Bangladeshi-British cultural practitioner.
  Fact 0.4: Eenasul Fateh is a magician.
  Fact 0.5: Eenasul Fateh is a live artist.
  Fact 0.6: Eenasul Fateh is a former international management consultant.
Sentence 1: Management consulting is the practice of helping organizations to improve their performance, operating primarily through the analysis of existing organizational problems and the development of plans for improvement.
  Fact 1.0: Management consulting is the practice of helping organizations improve their performance.
  Fact 1.1: Management consulting operates primarily through the analysis of existing organizational problems.
  Fact 1.2: Management consulting operates primarily through the development of plans for improvement.

Retrieved items:
<item 1>
Aladdin's Eatery
Aladdin's Eatery is a chain of franchised restaurants in the American Midwest and South-East, specializing in Lebanese cuisine.

 Adapted to American tastes, the sites are fast casual restaurants that also offer take out.
</item 1>

<item 2>
Lawrence A. Gordon
Lawrence A. Gordon is the EY Alumni Professor of Managerial Accounting and Information Assurance at the University of Maryland’s Robert H. Smith School of
Business.

 He is also an Affiliate Professor in the University of Maryland Institute for Advanced Computer Studies.

 Dr. Gordon earned his Ph.D. in Managerial Economics from Rensselaer Polytechnic Institute.

 An internationally known scholar in the areas of managerial accounting (often called management accounting) and cybersecurity economics, Dr. Gordon's research
focuses on such issues as economic aspects of information security (including cybersecurity or computer security), corporate performance measures, cost
management systems, and capital investments.

 He is the author of approximately 100 articles, published in such journals as The Accounting Review, ACM Transactions on Information and System Security,
Communications of the ACM, Journal of Financial and Quantitative Analysis, Journal of Computer Security, MIS Quarterly, Accounting, Organizations and Society,
and the Journal of Accounting and Public Policy.

 Dr. Gordon's current research emphasizes the importance of applying concepts from economics and managerial accounting to an information-based economy.

 Dr.
</item 2>

<item 3>
Eenasul Fateh
Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live
artist and former international management consultant.
</item 3>

<item 4>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 4>

<item 5>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 5>

<item 6>
Alex Algard
Alex Algard is an Internet entrepreneur.

 He is the founder and former CEO of CarDomain, the founder and former CEO of Whitepages.com, and the founder and current CEO of Hiya.
</item 6>

<item 7>
Abd Allah ibn al-Fadl
Little is known of his life, apart from what can be gleaned from manuscripts of his texts.

 He was a deacon and the grandson of a bishop.

 He received an excellent education in both Arabic and Greek, having studied Arabic grammar with the famous poet Abul ʿAla Al-Maʿarri.

 Several of his works and translations were commissioned by notables from Antioch and neighbouring cities in Muslim territory.
</item 7>

<item 8>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 8>

<item 9>
Alan Crosland
Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
</item 9>

<item 10>
Da'ud Abu al-Fadl
Da'ud Abu al-Fadl (1161–1242) was a Karaite Jewish physician who lived in the Ayyubid Egypt in the twelfth century CE.

 He born in Cairo in 1161 and died there about 1242.

 Having studied medicine under the Jewish physician Hibat Allah ibn Jami, and under Abu al-Fafa'il ibn Naqid, he became the court physician of the sultan
al-Malik al-'Adil Abu Bakr ibn Ayyub, the brother and successor of Saladin.

 He was also chief professor at the al-Nasiri Hospital at Cairo, where he had a great many pupils, among them being the historian ibn Abi Usaibiyyah.

 The latter declared that Abu al-Fadl was the most skillful physician of the time and that his success in curing the sick was miraculous.

 Abu al-Fadl was the author of an Arabic pharmacopoeia in twelve chapters, entitled "Aḳrabadhin", and treating chiefly of antidotes.
</item 10>
=== END ITEM id=c825a0091d50e6e7 ===

=== ITEM id=65d10c9a188a0a8f ===
Gold sentences and facts:
Sentence 0: Broughtonia is a genus of orchids (family Orchidaceae) native to the Bahamas and the Greater Antilles.
  Fact 0.0: Broughtonia is a genus of orchids.
  Fact 0.1: Broughtonia belongs to the family Orchidaceae.
  Fact 0.2: Broughtonia is native to the Bahamas.
  Fact 0.3: Broughtonia is native to the Greater Antilles.
Sentence 1: Perilla is a genus consisting of one major Asiatic crop species "P. frutescens" and a few recognized wild species in nature, belonging to the mint family, Lamiaceae.
  Fact 1.0: Perilla is a genus.
  Fact 1.1: Perilla consists of one major Asiatic crop species, P. frutescens.
  Fact 1.2: Perilla includes a few recognized wild species in nature.
  Fact 1.3: Perilla belongs to the mint family, Lamiaceae.

Retrieved items:
<item 1>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 1>

<item 2>
Monardella lanceolata
Monardella lanceolata is a species of flowering plant in the mint family known by the common names mustang mint and mustang monardella.

 It is native to the mountains of California and Baja California, where it grows in chaparral, woodland, rocky slopes, and often disturbed habitat types.
</item 2>

<item 3>
Thonningia
Thonningia is a monotypic genus of flowering plants in the family Balanophoraceae containing the single species Thonningia sanguinea.

 It is distributed throughout much of southern and western Africa, particularly the tropical regions.

 Common names for the plant include ground pineapple.

 A familiar plant to humans, it has an extremely long list of common names in many African languages.

 Many names are inspired by the resemblance of the plant's inflorescence to a pineapple or palm tree.

 Some of the names can be translated as pineapple of the bush (from Anyi), duiker's kolanut (from Igala), and crown of the ground (from Yoruba).
</item 3>

<item 4>
Perilla
Perilla is a genus consisting of one major Asiatic crop species "P. frutescens" and a few recognized wild species in nature, belonging to the mint family,
Lamiaceae.

 It encompasses several distinct varieties of Asian herb, seed, and vegetable crop, including "P. frutescens" (deulkkae) and "P. frutescens" var.

 "crispa" (shiso).

 The genus name "Perilla" is also a frequently employed common name ("perilla"), applicable to all varieties.

 Perilla varieties are cross-fertile and intra-specific hybridization occurs naturally.

 Some varieties are considered invasive.
</item 4>

<item 5>
Perilla frutescens
Perilla frutescens, commonly called perilla or Korean perilla, is a species of "Perilla" in the mint family Lamiaceae.

 It is an annual plant native to Southeast Asia and Indian highlands, and grown in the Korean peninsula, Southern China and India.

 Known as "deulkkae" (들깨 ) in Korean, the plant was introduced into Korea before the Unified Silla era, when it started to be widely cultivated.
</item 5>

<item 6>
Lamiaceae
or Labiatae is a family of flowering plants commonly known as the mint or deadnettle family.

 Many of the plants are aromatic in all parts and include widely used culinary herbs, such as basil, mint, rosemary, sage, savory, marjoram, oregano, hyssop,
thyme, lavender, and perilla.

 Some species are shrubs, trees (such as teak), or, rarely, vines.

 Many members of the family are widely cultivated, not only for their aromatic qualities but also their ease of cultivation, since they are readily propagated
by stem cuttings.

 Besides those grown for their edible leaves, some are grown for decorative foliage, such as "Coleus".

 Others are grown for seed, such as "Salvia hispanica" (chia), or for their edible tubers, such as "Plectranthus edulis", "Plectranthus esculentus",
"Plectranthus rotundifolius", and "Stachys affinis" (Chinese artichoke).
</item 6>

<item 7>
Broughtonia
Broughtonia is a genus of orchids (family Orchidaceae) native to the Bahamas and the Greater Antilles.

 The genus is abbreviated Bro in trade journals.
</item 7>

<item 8>
Shiso
Perilla frutescens var.

 "crispa", also called shiso ( , from Japanese シソ ) is a variety of species "Perilla frutescens" of the genus "Perilla", belonging to the mint family,
Lamiaceae.

 The plant occurs in red (purple-leaved) or green-leaved forms.

 Shiso is a perennial plant that may be cultivated as an annual in temperate climates, and occurs in both red- (or purple-) leaved and green forms.

 There are also frilly, ruffled-leaved forms called "chirimen-jiso" and forms that are red only on top, called "katamen-jiso".
</item 8>

<item 9>
Perilla ketone
Perilla ketone is a natural terpenoid that consists of a furan ring with a six-carbon side chain containing a ketone functional group.

 It is a colorless oil that is sensitive to oxygen, becoming colored upon standing.

 The ketone was identified in 1943 by Sebe as the main component of the essential oil of Perilla frutescens.

 Perilla ketone is present in the leaves and seeds of purple mint ("Perilla frutescens"), which is toxic to some animals.

 When cattle and horses consume purple mint when grazing in fields in which it grows, the perilla ketone causes pulmonary edema leading to a condition sometimes
called perilla mint toxicosis.
</item 9>

<item 10>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 10>
=== END ITEM id=65d10c9a188a0a8f ===

=== ITEM id=793cd2c9630d2a7e ===
Gold sentences and facts:
Sentence 0:  The album includes collaborations with Popcaan, Zhu, Leikeli47, Dreezy, Flume and Pell.
  Fact 0.0: The album includes a collaboration with Popcaan.
  Fact 0.1: The album includes a collaboration with Zhu.
  Fact 0.2: The album includes a collaboration with Leikeli47.
  Fact 0.3: The album includes a collaboration with Dreezy.
  Fact 0.4: The album includes a collaboration with Flume.
  Fact 0.5: The album includes a collaboration with Pell.
Sentence 1: Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of 2014, signed to Mind of a Genius Records and Columbia Records.
  Fact 1.0: Steven Zhu was born in 1989.
  Fact 1.1: Steven Zhu is known professionally as Zhu.
  Fact 1.2: Zhu is a Chinese American electronic musician.
  Fact 1.3: Zhu is a Chinese American singer.
  Fact 1.4: Zhu has been active since the beginning of 2014.
  Fact 1.5: Zhu is signed to Mind of a Genius Records.
  Fact 1.6: Zhu is signed to Columbia Records.

Retrieved items:
<item 1>
King Electric
King Electric is Chris Isepp and Peter Hartwig.

 They took their first musical attempts at the end of 1980s / beginning 1990s and first met each other at the upcoming and flourishing Viennese techno scene
back in '92.

 While Hartwig was touring with the well known noise-pop band the Basket Boys, Isepp made first attentions as Electro DJ & Producer.
</item 1>

<item 2>
Cherushii
Cherushii (Chelsea Faith Dolan) (September 14, 1983 – December 2, 2016) was an American electronic music producer, DJ, live performer, and radio host based in
San Francisco.

 Influenced by house music, Detroit techno, and disco, she released several recordings on the independent dance music label 100% Silk, including "Queen of
Cups", "Far Away So Close", and "Memory of Water", reissued in 2017.

 KQED named Cherushii's album "Meow Wolf’s Arcade Soundtracks: Wiggy’s Plasma Plex" one of the 10 best bay area albums of 2016.

 While Cherushii has received little recognition from mainstream media, her music is of enormous influence in the underground music scene.

 Fellow electronic musician Maria Minerva has called Cherushii criminally underrated.
</item 2>

<item 3>
Liao Changyong
Prof. Liao Changyong (; born October 25, 1968), sometimes referred in Western media as C. Y. Liao or Changyong Liao, is a Chinese operatic baritone and
academic.

 He won first prize in three different international competitions in 1996 and 1997: the Operalia, The World Opera Competition; the French International Toulouse
Singing Competition; and the Queen Sonja International Music Competition.

 While his performance career has mainly been in China, he has appeared as a guest artist with opera companies and orchestras internationally.

 He is the head of the voice department at the Shanghai Conservatory of Music.
</item 3>

<item 4>
Zhu (musician)
Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of
2014, signed to Mind of a Genius Records and Columbia Records.

 Until mid-2014, Zhu remained anonymous, asking to be judged by his music alone.

 His debut album, "Generationwhy", was released on 29 July 2016.
</item 4>

<item 5>
Vektroid
Ramona Andra Xavier (born August 19, 1992, known by her primary alias Vektroid) is an American electronic musician from Portland, Oregon.

 She has released music under her primary Vektroid alias, as well as others, such as dstnt, Laserdisc Visions, New Dreams Ltd., Macintosh Plus, Virtual
Information Desk, and PrismCorp Virtual Enterprises.

 Xavier played a prominent role in the popularization of the vaporwave subgenre with the release of her only studio album under her Macintosh Plus alias,
"Floral Shoppe", in 2011.

 The album helped popularize the vaporwave genre throughout the Internet.

 Since then, she has continued to release music through Bandcamp and other online platforms, and has recently collaborated with rapper Siddiq on two albums,
"Midnight Run" and "Palacio Del Rio" (both from 2016).
</item 5>

<item 6>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 6>

<item 7>
Kid Confucius
Kid Confucius were an eight-piece Australian band from Sydney, Australia.

 They formed in 2001 and have since played well over 400 live shows around the country, including major festivals as well as their own headline shows at venues
such as The Annandale Hotel and The Metro Theatre.

 In 2005 the band released a self-titled album and two singles "Words" and "Skintight."

 The album, a mish-mash of soul, hip-hop, pop and funk, received great critical acclaim, most notably from Rolling Stone who hailed the album as one of the
standout local releases of the year.

 "Words" also enjoyed some solid months of radio and TV play.

 The band released its second album Stripes in 2007 with three singles "Closer", "Last Straw" and "Moment".

 Stripes was the band's attempt at making a Detroit-era Motown soul album and it was quick to earn rave reviews from press around the country as well as a
publishing deal with Mushroom.

 "Moment" has enjoyed good radio play on triple j. Kid Confucius is set to release its third album in October 2008.

 The first single from this album, "Good Luck", is out now.
</item 7>

<item 8>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 8>

<item 9>
I Remember (AlunaGeorge album)
I Remember is the second studio album by English electronic music duo AlunaGeorge.

 It was released on 16 September 2016 by Island Records.

 The album includes collaborations with Popcaan, Zhu, Leikeli47, Dreezy, Flume and Pell.

 The album's lead single, "I'm in Control", was released in February 2016, featuring vocals from rapper Popcaan, and peaked at number 39 on the UK Singles
Chart.
</item 9>

<item 10>
Candy Chen
Candy Chen (陳斯亞 born January 3, 1993) is a dancer, actress, host, singer, rapper and a model.

 She is featured on the Taiwanese variety shows: Blackie's Teenage Club and Blackie Lollipop.

 Candy was a member of the Taiwanese girl group Hey Girl from 2010 to 2011, and a member of from 2013 to 2016.

 She is 1/8th Dutch.

 She changed her name to Chen Yi-Ling in July 2012 to follow her mother's maiden name.

 Her English name becomes Nina Chen.

 In November 2013, she changed her name to Candy Chen (English Name) and Chen Si Ya (Chinese Name).
</item 10>
=== END ITEM id=793cd2c9630d2a7e ===

=== ITEM id=212e3701c645fa44 ===
Gold sentences and facts:
Sentence 0: The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.
  Fact 0.0: The Long–Allen Bridge is a truss bridge.
  Fact 0.1: The Long–Allen Bridge is in Shreveport, Louisiana.
  Fact 0.2: The Long–Allen Bridge is named for Louisiana governor Huey Long.
  Fact 0.3: The Long–Allen Bridge is named for Louisiana governor Oscar K. Allen.
Sentence 1:  Opened in 1933, it carries U.S. Route 80 across the Red River.
  Fact 1.0: The Long–Allen Bridge opened in 1933.
  Fact 1.1: The Long–Allen Bridge carries U.S. Route 80 across the Red River.
Sentence 2: U.S. Route 80 (US 80) is an east–west United States Numbered Highway, much of which was once part of the early auto trail known as the Dixie Overland Highway.
  Fact 2.0: U.S. Route 80 (US 80) is an east–west United States Numbered Highway.
  Fact 2.1: Much of U.S. Route 80 was once part of the early auto trail known as the Dixie Overland Highway.
Sentence 3:  The highway's eastern terminus is in Tybee Island, Georgia, at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street, near the Atlantic Ocean.
  Fact 3.0: The eastern terminus of U.S. Route 80 is in Tybee Island, Georgia.
  Fact 3.1: The eastern terminus of U.S. Route 80 is at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street.
  Fact 3.2: The eastern terminus of U.S. Route 80 is near the Atlantic Ocean.

Retrieved items:
<item 1>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 1>

<item 2>
Broad Street Bridge (Rochester, New York)
Erie Canal: Second Genesee Aqueduct, also known as the Broad Street Aqueduct or Broad Street Bridge, is a historic stone aqueduct located at Rochester in Monroe
County, New York.

 It was constructed in 1836-1842 and originally carried the Erie Canal over the Genesee River.

 The overall length of the aqueduct including the wings and abutments is 800 ft .

 The aqueduct is 70 ft wide and has massive parapets on either side.

 It is one of four major aqueducts in the mid-19th century Erie Canal system.

 In 1927, a roadbed was added to carry automobile traffic and named Broad Street.

 It also carried a part of the Rochester Subway.
</item 2>

<item 3>
Rocky Creek Bridge No. 01089
Rocky Creek Bridge No. 01089, also known as Ben Jones Bridge, is a concrete highway bridge spanning Rocky Creek along the Pacific Ocean coast of the U.S. state
of Oregon.

 The bridge crosses a gorge near the creek's mouth, about 2 mi south of Depoe Bay in Lincoln County.

 Built in 1927, the bridge originally carried U.S. Route 101 (the Oregon Coast Highway) over the creek.

 After a stretch of the highway was relocated to make it straighter, the abandoned piece of the old highway became Otter Crest Loop Road, a bypass west of the
new highway.
</item 3>

<item 4>
U.S. Route 80
U.S. Route 80 (US 80) is an east–west United States Numbered Highway, much of which was once part of the early auto trail known as the Dixie Overland Highway.

 As the "0" in the route number indicates, it was originally a cross-country route, from the Pacific Ocean to the Atlantic Ocean.

 However, the entire segment west of Dallas, Texas, has been decommissioned in favor of various Interstate Highways and state highways.

 Currently, the highway's western terminus is at an interchange with Interstate 30 (I-30) on the Dallas–Mesquite, Texas city line.

 The highway's eastern terminus is in Tybee Island, Georgia, at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street, near the Atlantic Ocean.
</item 4>

<item 5>
Grand Central Parkway
The Grand Central Parkway (GCP) is a 14.61-mile (23.51 km) long parkway that stretches from the Triborough Bridge in New York City to Nassau County on Long
Island.

 At the Queens–Nassau border, it becomes the Northern State Parkway, which runs across the northern part of Long Island through Nassau County and into Suffolk
County, where it ends in Hauppauge.

 The westernmost stretch (from the Triborough Bridge to exit 4) also carries a short stretch of Interstate 278 (I-278).

 The parkway runs through Queens and passes the Cross Island Parkway, Long Island Expressway, LaGuardia Airport and Citi Field, home of the New York Mets.

 The parkway is designated New York State Route 907M (NY 907M), an unsigned reference route.

 Despite its name, the Grand Central Parkway was not named after Grand Central Terminal.
</item 5>

<item 6>
Long–Allen Bridge (Shreveport)
The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.

 Opened in 1933, it carries U.S. Route 80 across the Red River.

 It is also known as the Texas Street Bridge.
</item 6>

<item 7>
Newark Bay
Newark Bay is a tidal bay at the confluence of the Passaic and Hackensack Rivers in northeastern New Jersey.

 It is home to the Port Newark-Elizabeth Marine Terminal, the largest container shipping facility in Port of New York and New Jersey, the third largest and one
of the busiest in the United States.

 An estuary, it is periodically dredged to accommodate ocean-going ships.
</item 7>

<item 8>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 8>

<item 9>
Ocean To Ocean Bridge
The Ocean To Ocean Bridge is a through truss bridge spanning the Colorado River in Yuma, Arizona.

 Built in 1915, it was the first highway crossing of the lower Colorado and is the earliest example of a through truss bridge in Arizona.

 It is also the only example of a Pennsylvania truss within Arizona.

 Originally the bridge carried the transcontinental Ocean-to-Ocean Highway and later carried its successor, US 80 until a new bridge was built to the west in
1956.

 Between 1988 and 2001, the bridge was closed to vehicular traffic and only traversable by pedestrians and bicyclists.

 After a major restoration, the bridge was rehabilitated and reopened to vehicular traffic in 2002, with a re-dedication by the Quechan nation and Yuma Crossing
National Heritage Area.
</item 9>

<item 10>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 10>
=== END ITEM id=212e3701c645fa44 ===

=== ITEM id=bf906c15ffafd6a3 ===
Gold sentences and facts:
Sentence 0: Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.
  Fact 0.0: Eriogonum is the scientific name for a genus of flowering plants.
  Fact 0.1: The genus Eriogonum is in the family Polygonaceae.
Sentence 1: Aloinopsis is a genus of ice plants from South Africa.
  Fact 1.0: Aloinopsis is a genus of ice plants.
  Fact 1.1: The Aloinopsis ice plants come from South Africa.

Retrieved items:
<item 1>
Aizoaceae
Aizoaceae (the fig-marigold family) is a large family of dicotyledonous flowering plants containing 135 genera and about 1900 species.

 They are commonly known as stone plants or carpet weeds.

 They are often called vygies in South Africa and New Zealand.

 Species that resemble stones or pebbles are sometimes called mesembs.

 Several species are known as ice plants because of the glistening globular bladder cells covering their stems, fruit and leaves, "... they sparkle like ice
crystals."
</item 1>

<item 2>
Eriogonum diclinum
Eriogonum diclinum is a species of wild buckwheat known by the common name Jaynes Canyon buckwheat.

 It is native to the Klamath Mountains of northern California and southern Oregon, where it is an uncommon member of plant communities on serpentine soils.

 This is a small dioecious shrub forming low, thick mats rarely exceeding 20 centimeters in height.

 The curving, oval-shaped leaves are woolly and greenish gray in color, growing up to two centimeters long.

 The plant flowers in yellow to reddish rounded inflorescences, male plants producing clusters of staminate flowers, and female plants producing larger clusters
of pistillate flowers.
</item 2>

<item 3>
Eriogonum heracleoides
Eriogonum heracleoides (common names parsnipflower buckwheat, whorled buckwheat, and Wyeth buckwheat) has many flowering clusters that are cream or off-white in
color.

 Its usual habitat is rocky areas such as sagebrush deserts and Ponderosa pine forests.

 Parsnipflower buckwheat is in the genus "Eriogonum" and the family Polygonaceae which is a family of plants known as the "knotweed family".

 It inhabits much of the western part of the United States.
</item 3>

<item 4>
Aloinopsis
Aloinopsis is a genus of ice plants from South Africa.
</item 4>

<item 5>
Eriogonum crosbyae
Eriogonum crosbyae is a species of wild buckwheat known by the common name Crosby's buckwheat.

 It is native to southcentral Oregon and northwestern Nevada in the United States.

 Some treatments include plants in Montana and Idaho as members of this species.

 This plant was first discovered in the Guano Valley in Lake County, Oregon, in 1978 by Bureau of Land Management botanist Virginia Crosby, and it was named for
her in 1981.
</item 5>

<item 6>
Yucca aloifolia
Yucca aloifolia is the type species for the genus Yucca.

 Common names include aloe yucca, dagger plant, and Spanish bayonet.

 It grows in sandy soils, especially on sand dunes along the coast.

 "Yucca aloifolia" is native to the Atlantic and Gulf Coasts of the United States from southern Virginia south to Florida and west to the Texas Gulf Coast, to
Mexico along the Yucatán coast, and to Bermuda, and parts of the Caribbean.

 Normally Yucca aloifolia is grown in USDA zones 8 through 11.

 Yucca aloifolia is a popular landscape plant in beach areas along the lower East Coast from Virginia to Florida.
</item 6>

<item 7>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 7>

<item 8>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 8>

<item 9>
Conicosia
Conicosia is a genus of succulent plants in the ice plant family native to southern Africa.

 They are known commonly as narrow-leafed ice plants.

 These are relatively short-lived perennials with underground stems and tentacle-shaped, dull-pointed triangular leaves.

 They bear large tubular flowers often exceeding 10 centimeters in width, with up to 250 fringelike petals arranged in a ring around a center with hundreds of
stamens.

 The fruit is a capsule which opens when it gets wet, slowly releasing the hundreds of tiny seeds as they fall out of its drying flesh.
</item 9>

<item 10>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 10>
=== END ITEM id=bf906c15ffafd6a3 ===

=== ITEM id=3631daad13a0d09b ===
Gold sentences and facts:
Sentence 0: Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.
  Fact 0.0: Old Smokey is a euphemistic name for the state prison electric chair in New Jersey.
  Fact 0.1: The New Jersey state prison electric chair, called Old Smokey, is on display at the New Jersey State Police Museum.
Sentence 1:  The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
  Fact 1.0: The chair's most notorious victim was Richard Hauptmann.
  Fact 1.1: Richard Hauptmann was executed in the chair.
  Fact 1.2: Richard Hauptmann was found guilty of abducting and killing Charles Augustus Lindbergh, Jr.
  Fact 1.3: The abduction and killing of Charles Augustus Lindbergh, Jr. happened in 1932.
  Fact 1.4: Charles Augustus Lindbergh, Jr. was 20 months old.
  Fact 1.5: Charles Augustus Lindbergh, Jr. was the son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh.
  Fact 1.6: The case is referred to as the Lindbergh kidnapping (see Lindbergh kidnapping).
Sentence 2:  The Lindbergh kidnapping became known as "The Crime of the Century".
  Fact 2.0: The Lindbergh kidnapping became known as 'The Crime of the Century'.

Retrieved items:
<item 1>
Copsi
Osulf, however, only ruled as earl until that autumn, when he was killed by an outlaw he was tracking.
</item 1>

<item 2>
Smiley face murder theory
The Smiley face murder theory (variations include Smiley face murders, Smiley face killings, Smiley face gang, and others) is a theory advanced by two retired
New York City detectives, Kevin Gannon and Anthony Duarte, that a number of young men found dead in bodies of water across several Midwestern American states
over the last decade did not accidentally drown, as concluded by law enforcement agencies, but were victims of a serial killer or killers.

 The term "smiley face" became connected to the alleged murders when it was made public that the police had discovered graffiti depicting a smiley face near
locations where they think the killer dumped the bodies in at least a dozen of the cases.

 The response of law enforcement investigators and other experts to Gannon and Duarte's theory has been largely skeptical.
</item 2>

<item 3>
Convicted (1950 film)
Convicted is a 1950 American crime film noir directed by Henry Levin starring Glenn Ford and Broderick Crawford.

 It was the third Columbia Pictures film adaptation of the 1929 stage play "The Criminal Code" by Martin Flavin, following Howard Hawk's "The Criminal Code"
(1931) and John Brahm's "Penitentiary" (1938).
</item 3>

<item 4>
Old Smokey
Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.

 The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus
Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
</item 4>

<item 5>
Cordelia Botkin
Cordelia Botkin (1854 – March 7, 1910) was an American murderer who sent a box of poisoned candy to her ex-lover's wife.

 This was the first American prosecution for a crime which took place in two different jurisdictions, as Botkin had sent the poison from California, but it was
received in Delaware.
</item 5>

<item 6>
The Journalist and the Murderer
The Journalist and the Murderer is a study by Janet Malcolm about the ethics of journalism, published by Alfred A. Knopf/Random House in 1990.

 It is an examination of the professional choices that shape a work of non-fiction, as well as a rumination on the morality that underpins the journalistic
enterprise.

 The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss'
1983 book "Fatal Vision".
</item 6>

<item 7>
Johnny Jack Nounes
Johnny Jack Nounes, also known as the "Beau Brummell of Galveston", was an organized crime boss in Galveston, Texas, United States, during the early 1900s.

 He, with one-armed George Musey, led the Downtown Gang, one of the two gangs which controlled most of the Galveston Crime Syndicate until the early 1930s.

 They fought for control of the island against the rival Beach Gang led by Ollie Quinn and Dutch Voight.

 As the prohibition era began, his gang came to be one of the dominant forces in the Galveston Crime Syndicate.

 Galveston became the main port of entry for liquor supply in Texas and many parts of the Midwest.
</item 7>

<item 8>
Old Sparky
Old Sparky is the nickname of the electric chairs in Arkansas, Connecticut, Florida, Georgia, Illinois, Kentucky, Nebraska, New York, Ohio, Oklahoma, South
Carolina, Texas, Virginia, and West Virginia.

 Old Smokey was the nickname of the electric chairs used in New Jersey, Pennsylvania, and Tennessee.
</item 8>

<item 9>
Richard Hauptmann
Bruno Richard Hauptmann (November 26, 1899 – April 3, 1936) was a German-born carpenter who was convicted of the abduction and murder of the 20-month-old son of
aviator Charles Lindbergh and his wife Anne Morrow Lindbergh.

 The Lindbergh kidnapping became known as "The Crime of the Century".

 Hauptmann proclaimed his innocence to the end, but he was convicted of first degree murder and executed in 1936 in the electric chair at the New Jersey State
Prison.
</item 9>

<item 10>
The Last Mile (play)
The Last Mile is a Broadway play by John Wexley that ran for 289 performances from February 13, 1930 to October 1930 at the Sam H. Harris Theatre.

 It was produced by Herman Shumlin and staged by Chester Erskine.

 It is set in the death row wing of a prison.

 The lead role of John "Killer" Mears was first played by Spencer Tracy, and it was the role that bought him to the attention of Hollywood.

 It was later played for a time by Clark Gable.

 The play was adapted into a 1932 film starring Preston Foster and into a 1959 film starring Mickey Rooney.
</item 10>
=== END ITEM id=3631daad13a0d09b ===

=== ITEM id=8114d86eec3bb9b9 ===
Gold sentences and facts:
Sentence 0: of Montreal is an American experimental pop band from Athens, Georgia.
  Fact 0.0: of Montreal is an American band.
  Fact 0.1: of Montreal is an experimental pop band.
  Fact 0.2: of Montreal is from Athens, Georgia.
Sentence 1: Vib Gyor was a band that was formed in Leeds, England.
  Fact 1.0: Vib Gyor was a band.
  Fact 1.1: Vib Gyor was formed in Leeds, England.

Retrieved items:
<item 1>
Montreal Victorias
The Victoria Hockey Club of Montreal, Quebec, Canada was an early men's amateur ice hockey club.

 Its date of origin is ascribed to either 1874, 1877 or 1881, making it either the first or second organized ice hockey club after McGill University.

 The club played at its own rink, the Victoria Skating Rink in Montreal.

 The club was winners of the Stanley Cup in 1895 and held it except for a period in 1896 until 1899.

 The club remained amateur, splitting from the ranks of teams turned professional in 1908.

 The amateur hockey club was the first winner of the Allan Cup and continued in play until 1939 after its 65th season.

 The club often also fielded junior and intermediate teams.
</item 1>

<item 2>
The White EP (Vib Gyor EP)
The "White EP" is a promo EP from the Leeds band Vib Gyor.

 It was first sold on their UK tour with American band Transfer and was released through their My Space page and official web-site on 13 November 2007.

 It is the follow-up to their The Secret EP.

 It was recorded in Rockfield studios in South Wales and produced by Ken Thomas.
</item 2>

<item 3>
Fallen (Vib Gyor song)
"Fallen" was the first single from the Leeds band Vib Gyor.

 It was first given limited release in July 2006, and has attained critical acclaim and a large amount of radio play on both sides of the atlantic.
</item 3>

<item 4>
The Famines (band)
The Famines are a Canadian indie rock band formed in 2008 in Edmonton, Alberta, Canada now based in Montreal, Quebec, Canada.

 The two piece band uses a modern and minimalistic approach that draws comparison to mid 1970's protopunk and fuzzy garage rock.

The band name is meant to be a commentary on the continued feeling of emptiness and lacking in a society that is materially fulfilled.

 The band has two members, R. E. Biesinger on guitar and vocals, and Drew Demers on the drums.
</item 4>

<item 5>
Vib Gyor
Vib Gyor was a band that was formed in Leeds, England.

 The name was derived from the first letters of all the colours in the rainbow, in order from the shortest to longest wavelengths.
</item 5>

<item 6>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 6>

<item 7>
Of Montreal
of Montreal is an American experimental pop band from Athens, Georgia.

 It was founded by frontman Kevin Barnes in 1996, named after a failed romance between Barnes and a woman "of Montreal."

 The band is identified as part of the Elephant 6 collective.

 Throughout its existence, of Montreal's musical style has evolved considerably and drawn inspiration from numerous 1960s psychedelic pop acts.
</item 7>

<item 8>
The Gorgeous
The Gorgeous were a Canadian metalcore band formed in the summer of 2003 in Montreal.

 The band formed with Dali Shaw on guitar, Julien Brousseau on Bass and Miguel Shaw on drums.

 The three played a few shows without a vocalist and quickly recruited Jordon Daniel.
</item 8>

<item 9>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 9>

<item 10>
Bloc Party
Bloc Party are an English indie rock band, currently composed of Kele Okereke (lead vocals, rhythm guitar, keyboards, sampler), Russell Lissack (lead guitar,
keyboards), Justin Harris (bass guitar, keyboards, saxophones, backing vocals) and Louise Bartle (drums, percussion).

 Former members Matt Tong and Gordon Moakes left the band in 2013 and 2015 respectively.

 Their brand of music, whilst rooted in rock, retains elements of other genres such as electronica and house music.

 The band was formed at the 1999 Reading Festival by Okereke and Lissack.

 They went through a variety of names before settling on Bloc Party in 2003.

 Moakes joined the band after answering an advert in "NME" magazine, while Tong was picked via an audition.

 Bloc Party got their break by giving BBC Radio 1 DJ Steve Lamacq and Franz Ferdinand's lead singer, Alex Kapranos, a copy of their demo "She's Hearing Voices".
</item 10>
=== END ITEM id=8114d86eec3bb9b9 ===
