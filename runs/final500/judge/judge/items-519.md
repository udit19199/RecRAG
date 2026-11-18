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

=== ITEM id=9c0e9b6a09b2227d ===
Gold sentences and facts:
Sentence 0: Eryngium is a genus of flowering plants in the family Apiaceae.
  Fact 0.0: Eryngium is a genus of flowering plants.
  Fact 0.1: Eryngium is in the family Apiaceae.
Sentence 1: Arnebia is a genus of flowering plants in the family Boraginaceae.
  Fact 1.0: Arnebia is a genus of flowering plants.
  Fact 1.1: Arnebia is in the family Boraginaceae.

Retrieved items:
<item 1>
Ardisia
Ardisia (coralberry or marlberry) is a genus of flowering plants in the family Primulaceae.

 It was in the former Myrsinaceae family now recognised as the myrsine sub-family Myrsinoideae.

 They are distributed in the Americas, Asia, Australia, and the Pacific Islands, mainly in the tropics.

 There are 400 to 500 species.
</item 1>

<item 2>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 2>

<item 3>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 3>

<item 4>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 4>

<item 5>
Cyanobiont
Several cyanobionts involved with fungi and marine organisms also belong to the genera "Richelia, Calothrix, Synechocystis, Aphanocapsa" and "Anabaena", as well
as the species "Oscillatoria spongeliae".

 Although there are many documented symbioses between cyanobacteria and marine organisms, little is known about the nature of many of these symbioses.

 The possibility of discovering more novel symbiotic relationships is apparent from preliminary microscopic observations.
</item 5>

<item 6>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 6>

<item 7>
Vascular plant
Vascular plants (from Latin "vasculum": duct), also known as tracheophytes (from the equivalent Greek term "trachea") and also higher plants, form a large group
of plants (c. 308,312 accepted known species ) that are defined as those land plants that have lignified tissues (the xylem) for conducting water and minerals
throughout the plant.

 They also have a specialized non-lignified tissue (the phloem) to conduct products of photosynthesis.

 Vascular plants include the clubmosses, horsetails, ferns, gymnosperms (including conifers) and angiosperms (flowering plants).

 Scientific names for the group include Tracheophyta and Tracheobionta.
</item 7>

<item 8>
Arnebia
Arnebia is a genus of flowering plants in the family Boraginaceae.
</item 8>

<item 9>
Lophodermium
Lophodermium is a genus of fungi within the family Rhytismataceae.

 The genus contains 145 species and has a global distribution.

 Species of this genus are usually observed producing zone lines, conidiomata and ascomata on dead fallen leaves, but at least some are known to colonize living
leaves.

 In many cases they then live inside the colonized leaf as a symptomless endobiont, where they are regarded as detritivores utilising dead plant matter.

 In a few cases they may kill all or part of the leaf prematurely, and there is a substantial literature dealing with those species as plant pathogens.

 The genus infects many different plant families but with a notable concentration in the family Pinaceae; many "Lophodermium" species are restricted to a single
host genus (or even species), but some, particularly those infecting grasses, may infect several genera.

 Some are economically important plant pathogens, such as those that cause needlecast disease in European Black Pine, Scots Pine and Red Pine in forestry and
christmas tree plantations.
</item 9>

<item 10>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 10>
=== END ITEM id=9c0e9b6a09b2227d ===

=== ITEM id=725c614a5e639126 ===
Gold sentences and facts:
Sentence 0:  After some time spent in general practice, he became a Lecturer, Senior Lecturer and Principal Lecturer in animal behaviour at De Montfort University, and subsequently the University of Lincoln, where he has been Head of the Department of Biological Sciences.
  Fact 0.0: He spent some time in general practice.
  Fact 0.1: He became a Lecturer in animal behaviour at De Montfort University.
  Fact 0.2: He became a Senior Lecturer in animal behaviour at De Montfort University.
  Fact 0.3: He became a Principal Lecturer in animal behaviour at De Montfort University.
  Fact 0.4: He subsequently moved to the University of Lincoln.
  Fact 0.5: He has been Head of the Department of Biological Sciences at the University of Lincoln.
Sentence 1:  Individuals who have earned a Doctor of Philosophy degree may, in most jurisdictions, use the title of "Doctor" (often abbreviated "Dr") or, in non-English speaking countries, variants such as "Dr. phil."
  Fact 1.0: People who have earned a Doctor of Philosophy degree may, in most jurisdictions, use the title 'Doctor'.
  Fact 1.1: The title 'Doctor' is often abbreviated 'Dr'.
  Fact 1.2: In non-English speaking countries, people with a Doctor of Philosophy degree may use variants such as 'Dr. phil.'

Retrieved items:
<item 1>
Richard F. Wilson
Richard F. Wilson was selected to be Illinois Wesleyan University's 18th President on April 6, 2004 and took office on July 1, 2004.

 He was inaugurated on April 9, 2005.

 He had spent the previous 26 years at the University of Illinois at Urbana-Champaign, leaving as associate chancellor for development and vice president of the
University of Illinois Foundation.

 Mr. Wilson is a Ph.D. graduate of the University of Michigan.

 Wilson announced his intended retirement from Illinois Wesleyan, to take place around June 2015 or whenever a successor is named and transitioned.
</item 1>

<item 2>
William McGinnis
William "Bill" McGinnis, Ph.D. is a molecular biologist and professor of biology at the University of California San Diego.

 At UC San Diego he has also served as the Chairman of the Department of Biology from July 1998 - June 1999, as Associate Dean of the Division of Natural
Sciences from July 1, 1999 - June 2000, and as Interim Dean of the newly established Division of Biological Sciences from July 1, 2000 - February 1, 2001.

 Dr. McGinnis was appointed Dean of the Divisional Biological Sciences on July 1, 2013
</item 2>

<item 3>
David M. Sever
David M. Sever is an American herpetologist, histologist, anatomist and reproductive biologist.

 He has been a professor and department head in the Department of Biological Sciences at Southeastern Louisiana University since 2004, and held the Kenneth
Dyson Endowed Professorship in Biological Sciences from 2012 to 2015.

 He is well known for over 30 years of research on the secondary sexual characteristics of salamanders and more generally on comparative histoanatomy of the
urogenital systems of vertebrates.

 and was recognized as the 2013 Distinguished Herpetologist of the Year by the Herpetologists' League.
</item 3>

<item 4>
Lincoln University College, Malaysia
Lincoln University College, Petaling Jaya, Malaysia, was founded in the year in 2002 as Lincoln College and in 2011 Lincoln College was upgraded to Lincoln
University College.

 Lincoln University College is a private institution that provides higher education, approved by the Malaysian Qualifications Agency (MQA, National
Accreditation Board) and the Ministry of Higher Education.

 Situated in the cosmopolitan town of Petaling Jaya, the university college is close to the capital city, Kuala Lumpur.
</item 4>

<item 5>
Ekhard Salje
Ekhard Karl Hermann Salje, FRS (born 1946) is Professor of Mineralogy and Petrology and former Head of the Department of Earth Sciences, Cambridge University.
</item 5>

<item 6>
Daniel Mills (biologist)
Daniel Simon Mills, FRCVS (born 21 Aug 1966) is an English biologist and the UK's first Professor of Veterinary Behavioural Medicine based at the University of
Lincoln, United Kingdom.

 He attended St Edmund's College, Ware before studying at the University of Bristol where he received his BVSc degree and completed his Ph.D. degree in animal
behaviour from De Montfort University.

 After some time spent in general practice, he became a Lecturer, Senior Lecturer and Principal Lecturer in animal behaviour at De Montfort University, and
subsequently the University of Lincoln, where he has been Head of the Department of Biological Sciences.

 His research has focused on the assessment of emotion in companion animals and the use of semiochemicals to manage their problem behaviour (pheromonatherapy).
</item 6>

<item 7>
Richard W. Aldrich
Richard Warren Aldrich is an American neuroscientist who is currently the Karl Folkers Chair of Interdisciplinary Medical Research and Professor of Neurobiology
in the School of Biological Sciences at the University of Texas.

 Graduating from the University of Arizona with a BS in Biological Sciences (1975), Aldrich went on to earn a Ph. D. in Neuroscience from Stanford University
(1980).

 After completing a post-doctoral fellowship in physiology at Yale University under the direction of W. Knox Chandler and Charles F. Stevens, Aldrich was hired
to teach at Yale, instructing in the Department of Molecular Neurobiology.

 In 1985, Aldrich returned to Stanford to teach neurobiology and physiology, eventually serving as Chair of the Department of Molecular and Cellular Physiology
from 2001 to 2004.

 He was an Investigator in the Howard Hughes Medical Institute from 1990 until 2006.

 Aldrich's research has focused on the "molecular mechanisms of ion channel function and their role in electrical signaling."

 Aldrich has served as the President of the Society of General Physiologists and the Biophysical Society.
</item 7>

<item 8>
Ekhard Salje: Cambridge mineralogist and former Earth Sciences head. Peter Doherty: Australian veterinary immunologist; 1995 Lasker, 1996 medicine Nobel with
Zinkernagel, 1997 Australian of the Year. Antje Boetius: German geomicrobiologist; 2009 Leibniz laureate, studies seabed methane-oxidizing microbes and climate.
Brian Josephson: Welsh physicist; 1973 physics Nobel for the Joseph
</item 8>

<item 9>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 9>

<item 10>
Wale Adebanwi
Professor Wale Adebanwi, (born 1969), is a Nigerian - born first Black African Rhodes Professor at Oxford University
</item 10>
=== END ITEM id=725c614a5e639126 ===

=== ITEM id=4446c6026f4f43ee ===
Gold sentences and facts:
Sentence 0:  The film concentrated on Ellsworth "Bumpy" Johnson (Laurence Fishburne), Dutch Schultz (Tim Roth), and Lucky Luciano (Andy García).
  Fact 0.0: The film concentrated on Ellsworth "Bumpy" Johnson.
  Fact 0.1: Laurence Fishburne played Ellsworth "Bumpy" Johnson in the film.
  Fact 0.2: The film concentrated on Dutch Schultz.
  Fact 0.3: Tim Roth played Dutch Schultz in the film.
  Fact 0.4: The film concentrated on Lucky Luciano.
  Fact 0.5: Andy García played Lucky Luciano in the film.
Sentence 1:  Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.
  Fact 1.0: Lucky Luciano is considered the father of modern organized crime in the United States.
  Fact 1.1: Lucky Luciano is considered the father of modern organized crime in the United States for establishing the first Commission.
  Fact 1.2: Lucky Luciano established the first Commission.

Retrieved items:
<item 1>
Illtown
Dante and his girlfriend Micky run a very profitable drug operation in a seaside town, aided and abetted by a host of teens who sell the smack at discos around
town, as well as by Lucas, a corrupt cop who's on the take.

 Their downfall comes when they suspect one of the boys, Pep, of ripping them off, and his accidental death causes disloyalty among the teens, who suspect Dante
offed them.

 All of this is perfect for the return of Gabriel, a one-time partner of Dante, who has just been released from jail, and has an almost angelic demeanor and the
certainty that he can fix everyone's lives.
</item 1>

<item 2>
Frank Colacurcio
Francis Colacurcio, Sr. (June 18, 1917 – July 2, 2010) was an American businessman and boss of the Seattle crime family known for running strip clubs in
Seattle, Washington.

 He gained notoriety as a subject of ongoing federal investigations into organized crime in the city and was suspected of being an organized crime boss.
</item 2>

<item 3>
Hoodlum (soundtrack)
Hoodlum is the soundtrack to the 1997 American crime drama film "Hoodlum".

 It was released on August 12, 1997 by Interscope Records and consisted of a blend of hip hop and R&B music.

 The soundtrack peaked at 94 on the "Billboard" 200 and 23 on the Top R&B Albums and contained Mobb Deep's single "Hoodlum" which went to 29 on the Hot Rap
Singles.
</item 3>

<item 4>
Lucky Luciano
Charles "Lucky" Luciano ( ; born Salvatore Lucania November 24, 1897 – January 26, 1962) was an Italian-American mobster and crime boss.

 Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.

 He was also the first official boss of the modern Genovese crime family.

 He was, along with his associates, instrumental in the development of the National Crime Syndicate.
</item 4>

<item 5>
St. John Cemetery (Queens)
St. John Cemetery is an official Roman Catholic burial ground located in Middle Village in the Queens borough of New York City.

 It is one of nine official Roman Catholic burial grounds in the New York Metropolitan Area.

 St. John, along with St. Charles/Resurrection Cemeteries in Farmingdale, Long Island, is among the largest cemeteries in New York State.

 Since its opening, St. John has been the resting place of various famous and infamous people in New York City society, such as Mario Cuomo (1932–2015),
Governor of the state of New York from 1983–1995, John F. Hylan (1868–1936), mayor of the city of New York from 1918–1925, Geraldine Ferraro (1935–2011), the
first female Vice Presidential candidate representing a major American political party, Lucky Luciano (1897–1962), considered the father of modern organized
crime in the United States, and John J. Gotti (1940–2002), the head of the New York City based Gambino crime family from 1985–2002.
</item 5>

<item 6>
James Marcello
James J. Marcello (born December 13, 1943), also known variously as Little Jimmy, Jimmy Light and as Jimmy the Man Marcello, is an imprisoned crime boss who was
a front boss for the Chicago Outfit criminal organization in the 1980s, 1990s and early 2000s.

 Organized crime observers identified Marcello as a figurehead during that period while the organization's day-to-day operations actually were run by John “No
Nose” DiFronzo, Joseph “Joey the Clown” Lombardo, Joseph “Joe the Builder” Andriacchi and Angelo J. LaPietra.

 Marcello is the Cousin of Michael Hayes Marcello, and a cousin of Robert Urbinatti.

 He is also the father of Rocco Marcello.
</item 6>

<item 7>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 7>

<item 8>
Hoodlum (film)
Hoodlum is a 1997 American crime drama film that gives a fictionalized account of the gang war between the Italian/Jewish mafia alliance and the Black gangsters
of Harlem that took place in the late 1920s and early 1930s.

 The film concentrated on Ellsworth "Bumpy" Johnson (Laurence Fishburne), Dutch Schultz (Tim Roth), and Lucky Luciano (Andy García).
</item 8>

<item 9>
Ang Utol Kong Hoodlum
Ang Utol Kong Hoodlum (lit.

 "My Brother is a Hoodlum") is a Filipino drama series developed for TV5 created by Deo J. Fajardo.

 It stars JC de Vera and Jasmine Curtis-Smith.

 It is a remake of the original movie where Robin Padilla played the role of Ben, a hoodlum, and Vina Morales, as Bing.

 The movie was first released in 1991, then a sequel entitled "Miss na Miss Kita: Ang Utol Kong Hoodlum 2" was made the following year.

 It is produced by Vic Del Rosario Jr., and Manuel V. Pangilinan and it also marks as the first primetime series produced by Viva Television for TV5 after a
decade.
</item 9>

<item 10>
Joseph Todaro Jr.
Joseph "Big Joe" Todaro Jr. (born 1945 or 1946) is a Buffalo, New York businessman and former organized crime figure involved in labor racketeering,
loansharking, illegal gambling, narcotics, and murder for hire.

 Joe Todaro Jr. became a business agent for the Laborers' International Union of North America (LIUNA) Local 210.

 In 1984, Joe Todaro Jr. allegedly became the underboss of the Buffalo crime family after his father became the new head of the crime family, upon the
retirement of his predecessor Samuel "Sam the Farmer" Frangiamore.

 In 1990, Joe Todaro Jr. resigned as business agent following investigations on the local's alleged ties to organized crime.

 Outside of organized crime, Todaro operates La Nova Pizzeria, a popular pizza restaurant in Buffalo.
</item 10>
=== END ITEM id=4446c6026f4f43ee ===

=== ITEM id=4514a6ee89d1aa04 ===
Gold sentences and facts:
Sentence 0: Chalk and Cheese was a short lived sitcom television show, featuring Michael Crawford and Robin Hawdon.
  Fact 0.0: Chalk and Cheese was a short-lived sitcom television show
  Fact 0.1: Chalk and Cheese featured Michael Crawford
  Fact 0.2: Chalk and Cheese featured Robin Hawdon
Sentence 1: Michael Patrick Smith, {'1': ", '2': ", '3': ", '4': "} (born 19 January 1942), known as Michael Crawford, is an English actor, comedian and singer.
  Fact 1.0: Michael Patrick Smith is known as Michael Crawford
  Fact 1.1: Michael Patrick Smith was born 19 January 1942
  Fact 1.2: Michael Crawford is English
  Fact 1.3: Michael Crawford is an actor
  Fact 1.4: Michael Crawford is a comedian
  Fact 1.5: Michael Crawford is a singer

Retrieved items:
<item 1>
Dan Clark
Daniel Gregory Clark (born 3 July 1976 in Bromley, London) is an English actor, comedian, writer, director, and singer.

 He is best known for playing Don Danbury on the BBC Three sitcom "How Not to Live Your Life", which he also wrote, co-produced, and sometimes directed.

 He has been a regular on the British comedy scene as both a sketch and stand-up comedian.
</item 1>

<item 2>
Chalk and Cheese
Chalk and Cheese was a short lived sitcom television show, featuring Michael Crawford and Robin Hawdon.
</item 2>

<item 3>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 3>

<item 4>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 4>

<item 5>
Richard Cheese
Richard Cheese & Lounge Against the Machine is a cover band and comedy act, performing popular songs in a lounge/swing style, reminiscent of Frank Sinatra, Dean
Martin, and Tony Bennett.

 Lounge singer Richard Cheese is a character created and portrayed by Los Angeles-based actor/comedian Mark Jonathan Davis.
</item 5>

<item 6>
John Altman (actor)
John Clarkson Stewart (born 2 March 1952), known as John Altman, is an English actor and singer, perhaps best known for playing "Nasty" Nick Cotton in the
popular BBC soap opera "EastEnders".

 He was among the show's original cast members appearing in the very first episode in February 1985 and appeared on the show on and off as a recurring
character.

 His character was killed off in the 30th anniversary episode of the show which aired in February 2015.

 Altman has also appeared in several other television series and appeared in many stage productions.

 In 2010 he became the new frontman of the band Heavy Metal Kids following the death of former frontman Gary Holton in 1985.
</item 6>

<item 7>
A Cream Cracker under the Settee
"A Cream Cracker Under The Settee" is a dramatic monologue written by Alan Bennett in 1987 for television, as part of his "Talking Heads" series for the BBC.

 The series became very popular, moving onto BBC Radio, international theatre, becoming one of the best-selling audio book releases of all time and included as
part of both the A-level and GCSE English syllabus.

 It was the sixth and final episode of the first series of "Talking Heads".
</item 7>

<item 8>
Jonathan Routh
Jonathan Routh, born John Reginald Surdeval Routh, (1927–2008) co-starred in the British version of the television show "Candid Camera" (1960–67) and co-starred
with Germaine Greer and Kenny Everett in a later attempt at a revival, "Nice Time" (1968).

 He published a number of humorous books, and also painted for many years.
</item 8>

<item 9>
Roger Kitter
Roger Kitter (20 October 1949 – 3 January 2015) was an English actor best known for playing Captain Alberto Bertorelli in series 7 of the British sitcom TV
series "'Allo 'Allo!

".

 He had previously appeared weekly with Lulu throughout the 10-week run of her 1973 BBC1 series "It's Lulu".

 With Kaplan Kaye he also recorded a song "Chalk Dust – the Umpire Strikes Back" using the moniker 'The Brat'.

 Released on the Hansa label, it entered the UK Singles Chart on 10 July 1982; it reached a peak of number 19, and remained in the chart for 8 weeks.

 The song was a Top 10 hit in the Netherlands, Belgium and South Africa, and lampooned John McEnroe complaining about line calls in tennis ("The ball's in,
everyone can see that the ball's in!")

.

 He was also an impressionist, who provided the voice of Tommy Cooper in the Lego 'Kipper' advertisement.
</item 9>

<item 10>
Ken Barrie
Leslie Hulme (9 January 1933 – 29 July 2016), known professionally as Ken Barrie, was an English voice actor and singer best known for narrating, and singing
the theme tune of the BBC television programmes "Postman Pat" and "Charlie Chalk".

 He was also responsible for providing the voices of several of the series' characters.
</item 10>
=== END ITEM id=4514a6ee89d1aa04 ===

=== ITEM id=2e7d8a2ed5e876d9 ===
Gold sentences and facts:
Sentence 0:  It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts Mika Brzezinski and Willie Geist.
  Fact 0.0: The program described features former GOP representative Joe Scarborough.
  Fact 0.1: Joe Scarborough provides enterprise reporting on the program.
  Fact 0.2: Joe Scarborough provides discussion on the news of the day on the program.
  Fact 0.3: The program is in a panel format.
  Fact 0.4: Mika Brzezinski is a co-host of the program.
  Fact 0.5: Willie Geist is a co-host of the program.
Sentence 1: Charles Joseph Scarborough ( ; born April 9, 1963) is an American cable news and talk radio host.
  Fact 1.0: Charles Joseph Scarborough was born on April 9, 1963.
  Fact 1.1: Charles Joseph Scarborough is an American cable news host.
  Fact 1.2: Charles Joseph Scarborough is an American talk radio host.

Retrieved items:
<item 1>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 1>

<item 2>
Talk radio
Talk radio is a radio format containing discussion about topical issues and consisting entirely or almost entirely of original spoken word content rather than
outside music.

 Jason Culpepper is highly recognized as the father of talk radio.

 Jason a Jacksonville native rose to social prominence during the World War utilizing talk radio to warn french and american troops of German U boats.

 Most shows are regularly hosted by a single individual, and often feature interviews with a number of different guests.

 Talk radio typically includes an element of listener participation, usually by broadcasting live conversations between the host and listeners who "call in"
(usually via telephone) to the show.

 Listener contributions are usually screened by a show's producers in order to maximize audience interest and, in the case of commercial talk radio, attract
advertisers.

 Generally, the shows are organized into segments, each separated by a pause for advertisements; however, in public or non-commercial radio, music is sometimes
played in place of commercials to separate the program segments.

 Variations of talk radio include conservative talk, hot talk, liberal talk (increasingly known as progressive talk) and sports talk.
</item 2>

<item 3>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 3>

<item 4>
Mark Hinkle
Mark Hinkle (born January 28, 1951) is an American libertarian activist and businessperson.

 He was the National Chairman of the United States Libertarian Party until 2012.

 He was elected by the delegates of the 2010 Libertarian National Convention in St. Louis, Missouri on May 30, 2010.
</item 4>

<item 5>
Ronn Owens
Ronn Owens (born Ronald Lowenstein on October 17, 1945, in New York City), is an American talk radio host.

 Owens was the sole remaining weekday talk radio host on KGO in San Francisco after KGO was reorganized as an all-news station in December 2011.
</item 5>

<item 6>
Mike Hogan (sportscaster)
Mike Hogan (born 1963 in Kingston, Ontario) is a Canadian sportscaster who is a sports talk radio host on TSN Radio 1050 in Toronto.

 He also serves as the play-by-play voice for the Toronto Argonauts of the Canadian Football League for that station's gameday broadcasts.

 Hogan was formerly a long time sports talk radio host on Toronto station CJCL (AM), branded "The Fan 590", until he was relieved of those duties on June 24,
2010.

 On 2017 Oscar Vanity fair party 1st live broadcast Mike Hogan was the podcast hosts.
</item 6>

<item 7>
David Gold (talk radio host)
David Gold is an American conservative talk radio host.

 Gold, who has been a talk radio host since the mid-1970s, one of the first U.S. conservative talk radio hosts.

 His philosophy has been characterized as in the conservative/libertarian vein.
</item 7>

<item 8>
Sean Hannity
Sean Patrick Hannity (born December 30, 1961) is an American talk show host, author, and conservative political commentator.

 Hannity is the host of "The Sean Hannity Show", a nationally syndicated talk radio show.

 He also hosts a cable news show, "Hannity", on Fox News Channel.
</item 8>

<item 9>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 9>

<item 10>
Joe Scarborough
Charles Joseph Scarborough ( ; born April 9, 1963) is an American cable news and talk radio host.

 He is currently the co-host of "Morning Joe" on MSNBC, and previously hosted "Scarborough Country" on the same channel.

 Scarborough was previously a lawyer and a politician, and served in the United States House of Representatives from 1995 to 2001 as a Republican from the 1st
district of Florida.
</item 10>
=== END ITEM id=2e7d8a2ed5e876d9 ===

=== ITEM id=cb97c1735743408d ===
Gold sentences and facts:
Sentence 0: Ecopack Ltd is a PET bottles blowing company in Pakistan.
  Fact 0.0: Ecopack Ltd is a PET bottles blowing company.
  Fact 0.1: Ecopack Ltd is located in Pakistan.
Sentence 1: Polyethylene terephthalate (sometimes written poly(ethylene terephthalate)), commonly abbreviated PET, PETE, or the obsolete PETP or PET-P, is the most common thermoplastic polymer resin of the polyester family and is used in fibers for clothing, containers for liquids and foods, thermoforming for manufacturing, and in combination with glass fiber for engineering resins.
  Fact 1.0: Polyethylene terephthalate is also written as poly(ethylene terephthalate).
  Fact 1.1: Polyethylene terephthalate is commonly abbreviated as PET.
  Fact 1.2: Polyethylene terephthalate is commonly abbreviated as PETE.
  Fact 1.3: Polyethylene terephthalate is sometimes abbreviated as PETP, which is obsolete.
  Fact 1.4: Polyethylene terephthalate is sometimes abbreviated as PET-P, which is obsolete.
  Fact 1.5: Polyethylene terephthalate is the most common thermoplastic polymer resin of the polyester family.
  Fact 1.6: Polyethylene terephthalate is used in fibers for clothing.
  Fact 1.7: Polyethylene terephthalate is used in containers for liquids and foods.
  Fact 1.8: Polyethylene terephthalate is used in thermoforming for manufacturing.
  Fact 1.9: Polyethylene terephthalate is used in combination with glass fiber for engineering resins.

Retrieved items:
<item 1>
Two-liter bottle
The two-liter bottle is a common container for soft drinks.

 These bottles are produced from polyethylene terephthalate, also known as PET plastic, using the blow molding process.

 Bottle labels consist of a printed, tight-fitted plastic sleeve.

 A resealable screw-top allows the contents to be used at various times while retaining carbonation.
</item 1>

<item 2>
BoPET
BoPET (Biaxially-oriented polyethylene terephthalate) is a polyester film made from stretched polyethylene terephthalate (PET) and is used for its high tensile
strength, chemical and dimensional stability, transparency, reflectivity, gas and aroma barrier properties, and electrical insulation.
</item 2>

<item 3>
Ecopack
Ecopack Ltd is a PET bottles blowing company in Pakistan.
</item 3>

<item 4>
Xenoy
Xenoy is a blend of plastics with many industrially-useful properties.

 It is typically polyester (polybutylene terephthalate, PBT, or polyethylene terephthalate, PET) and polycarbonate (PC): it is often labeled PBT+PC or PET+PC.
</item 4>

<item 5>
Polytrimethylene terephthalate
Polytrimethylene terephthalate (PTT), is a polyester synthesized and patented in 1941.

 It is produced by a method called condensation polymerization or transesterification.

 The two monomer units used in producing this polymer are: 1,3-propanediol and terephthalic acid or dimethyl terephthalate.

 Similar to polyethylene terephthalate, the PTT is used to make carpet fibers.
</item 5>

<item 6>
Polyethylene 2,5-furandicarboxylate
Polyethylene 2,5-furandicarboxylate, also named poly(ethylene 2,5-furandicarboxylate), polyethylene furanoate and poly(ethylene furanoate) and generally
abbreviated as PEF, is a polymer that can be produced by polycondensation of 2,5-furandicarboxylic acid (FDCA) and monoethylene glycol (MEG).

 As an aromatic polyester from ethylene glycol it is a chemical analogue of polyethylene terephthalate (PET) and polyethylene naphthalate (PEN).

 PEF has been described in (patent) literature since 1951, but has gained renewed attention since the US department of energy proclaimed its building block,
FDCA, as a potential bio-based replacement for purified terephthalic acid (PTA) in 2004.
</item 6>

<item 7>
Polycyclohexylenedimethylene terephthalate
Polycyclohexylenedimethylene terephthalate (PCT) is a thermoplastic polyester formed from the polycondensation of terephthalic acid and cyclohexanedimethanol.

Its chemical structure is similar to that of polyethylene terephthalate (PET), with which it shares properties like dimensional stability and chemical
resistance.

 PCT is also particularly resistant to high temperatures and hydrolysis.

 The melting point is 545 °F (285 °C).

 Common brand names are "Thermx" (Ticona), "Eastar" (Eastman) and "SkyPURA" (SK Chemicals).
</item 7>

<item 8>
Krones
Krones AG is a German packaging and bottling machine manufacturer.

 It is the world's leading manufacturer of lines for filling beverages in plastic and glass bottles or beverage cans.

 The company manufactures stretch blow-moulding machines for producing polyethylene terephthalate (PET) bottles, plus fillers, labellers, bottle washers,
pasteurisers, inspectors, packers and palletisers.

 This product portfolio is complemented by material flow systems and process technology for producing beverages, plus syrup kitchens, for clients like
breweries, dairies and soft-drink companies.
</item 8>

<item 9>
Polyethylene terephthalate
Polyethylene terephthalate (sometimes written poly(ethylene terephthalate)), commonly abbreviated PET, PETE, or the obsolete PETP or PET-P, is the most common
thermoplastic polymer resin of the polyester family and is used in fibers for clothing, containers for liquids and foods, thermoforming for manufacturing, and
in combination with glass fiber for engineering resins.
</item 9>

<item 10>
Polyethylene naphthalate
Polyethylene naphthalate (poly(ethylene 2,6-naphthalate or PEN) is a polyester with good barrier properties (even better than Polyethylene terephthalate).

 Because it provides a very good oxygen barrier, it is particularly well-suited for bottling beverages that are susceptible to oxidation, such as beer.

 It is also used in making high performance sailcloth.

 It also has been found to show supreme scintillation properties and is expected to replace classic plastic scintillators.
</item 10>
=== END ITEM id=cb97c1735743408d ===

=== ITEM id=45e5ba0a664860cf ===
Gold sentences and facts:
Sentence 0: War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor at Kean University, and Albie Hecht.
  Fact 0.0: War/Dance is a 2007 American documentary film.
  Fact 0.1: War/Dance was written by Sean Fine and Andrea Nix Fine.
  Fact 0.2: War/Dance was directed by Sean Fine and Andrea Nix Fine.
  Fact 0.3: War/Dance was produced by Shine Global's Susan MacLaury.
  Fact 0.4: Susan MacLaury is a professor at Kean University.
  Fact 0.5: War/Dance was produced by Albie Hecht.
Sentence 1: The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy government and Nazi Germany during World War II.
  Fact 1.0: The Sorrow and the Pity is a documentary film.
  Fact 1.1: The Sorrow and the Pity's French title is Le Chagrin et la Pitié.
  Fact 1.2: The Sorrow and the Pity is in two parts.
  Fact 1.3: The Sorrow and the Pity was released in 1969.
  Fact 1.4: The Sorrow and the Pity was made by Marcel Ophüls.
  Fact 1.5: The Sorrow and the Pity is about the collaboration between the Vichy government and Nazi Germany.
  Fact 1.6: That collaboration occurred during World War II.

Retrieved items:
<item 1>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 1>

<item 2>
In the Mirror of Maya Deren
The documentary In the Mirror of Maya Deren ("Im Spiegel der Maya Deren", 2002) is a film about avant garde filmmaker Maya Deren (1917-1961) by Austrian film
maker Martina Kudlacek.

 It is based on the biography "The Legend of Maya Deren".
</item 2>

<item 3>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 3>

<item 4>
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 4>

<item 5>
The Great War (documentary)
The Great War is a 26-episode documentary series from 1964 on the First World War.

 The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian
Broadcasting Commission.

 The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.

 Each episode is  40 minutes long.
</item 5>

<item 6>
Polio v/s Polio victims
Polio v/s Polio Victims is the National Award Winning Documentary film.

 The documentary won "Best Motivational/ Educational Film" at the 56th National Film Awards.

 The documentary follows a group of polio victims who took to the streets in 2008 to spread the message of the Pulse Polio Campaign.

 The film follows them as they go door to door, in various slums in Mumbai, to spread awareness about the upcoming "Pulse Polio Day".
</item 6>

<item 7>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 7>

<item 8>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 8>

<item 9>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 9>

<item 10>
The Shame of a City
The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.

 Filmmaker Tigre Hill chronicles the 2003 Philadelphia mayoral race between Democrat incumbent mayor John Street and Republican challenger Sam Katz.

 Early polls showed Katz with a small lead in this predominantly Democratic city but twenty-seven days before the election, an FBI bug was found in the mayor’s
office.

 The discovery at first seemed like a death knell to the Street campaign and a near certain victory for Katz.

 Yet this prediction was proven wrong when Street and his supporters successfully polarized the campaign by leveling accusations of instituational racial
prejudice and playing on historical skepticism of the Republican-controlled federal government.

 As a result, Street won re-election by a sixteen-point margin.
</item 10>
=== END ITEM id=45e5ba0a664860cf ===

=== ITEM id=00211e070a78eca0 ===
Gold sentences and facts:
Sentence 0: The Doris Bither case, also known as the Entity hauntings, was an alleged haunting which occurred in 1974 at Culver City, California where a woman named Doris Bither alleged the ghosts of three Asian men were raping her.
  Fact 0.0: The Doris Bither case is also known as the Entity hauntings.
  Fact 0.1: The Doris Bither case was an alleged haunting.
  Fact 0.2: The alleged haunting occurred in 1974.
  Fact 0.3: The alleged haunting occurred at Culver City, California.
  Fact 0.4: Doris Bither alleged that the ghosts of three Asian men were raping her.
Sentence 1: The Entity is a 1982 American horror film directed by Sidney J. Furie and written by Frank De Felitta, who adapted his 1978 novel of the same name.
  Fact 1.0: The Entity is a 1982 American horror film.
  Fact 1.1: The Entity was directed by Sidney J. Furie.
  Fact 1.2: The Entity was written by Frank De Felitta.
  Fact 1.3: Frank De Felitta adapted his 1978 novel of the same name for The Entity.
Sentence 2:  Like the novel, the film is based on the 1974 Doris Bither case.
  Fact 2.0: The film is based on the 1974 Doris Bither case.
  Fact 2.1: The novel is also based on the 1974 Doris Bither case.

Retrieved items:
<item 1>
Doris Bither case
The Doris Bither case, also known as the Entity hauntings, was an alleged haunting which occurred in 1974 at Culver City, California where a woman named Doris
Bither alleged the ghosts of three Asian men were raping her.

 The case inspired Frank De Felitta's 1978 book "The Entity" which was made into a 1982 film of the same name starring Barbara Hershey.
</item 1>

<item 2>
Doctor Blood's Coffin
Doctor Blood's Coffin is a 1961 British horror film directed by Sidney J. Furie.

 Kieron Moore stars as a mad scientist who reanimates the dead.
</item 2>

<item 3>
Sheila Levine Is Dead and Living in New York
Sheila Levine Is Dead and Living in New York is a 1975 film directed by Sidney J. Furie, about a shy young woman who moves to New York City and falls in love
with the boyfriend of her extroverted roommate.

 The film was co-written by Kenny Solms and Gail Parent, and based on her novel.

 The film was shot on location in New York City.
</item 3>

<item 4>
The Snake Woman
The Snake Woman is a low budget 1961 British horror film directed by Sidney J. Furie and starring Susan Travers, John McCarthy.
</item 4>

<item 5>
The Entity
The Entity is a 1982 American horror film directed by Sidney J. Furie and written by Frank De Felitta, who adapted his 1978 novel of the same name.

 It stars Barbara Hershey as a woman who is raped and tormented by an invisible assailant.

 Despite being filmed and planned for a release in 1981, the movie was not released in worldwide theaters until September 1982 and February 1983 in the United
States.

 Like the novel, the film is based on the 1974 Doris Bither case.
</item 5>

<item 6>
Cannibal Holocaust
Cannibal Holocaust is a 1980 Italian cannibal horror film directed by Ruggero Deodato from a screenplay by Gianfranco Clerici.

 It stars Carl Gabriel Yorke, Robert Kerman, Francesca Ciardi, Perry Pirkanen, and Luca Barbareschi.

 Influenced by the works of Mondo director Gualtiero Jacopetti, the film was inspired by Italian media reporting of Red Brigade terrorism.

 The coverage included news reports Deodato believed to be staged, an idea which became an integral aspect of the film's story.

 "Cannibal Holocaust" was filmed primarily in the Amazon rainforest of Colombia with indigenous tribes interacting with American and Italian actors.
</item 6>

<item 7>
Shadows of the Night (film)
Shadows of the Night is a 1928 American drama silent film directed by D. Ross Lederman and written by Robert E. Hopkins and D. Ross Lederman.

 The film stars Flash the Dog, Lawrence Gray, Louise Lorraine, Warner Richmond and Tom Dugan.

 It was released on October 26, 1928, by Metro-Goldwyn-Mayer.
</item 7>

<item 8>
The Rocky Horror Picture Show
The Rocky Horror Picture Show is a 1975 British-American musical screwball comedy horror film by 20th Century Fox produced by Lou Adler and Michael White and
directed by Jim Sharman.

 The screenplay was written by Sharman and actor Richard O'Brien, who appears in the film, which is based on the 1973 musical stage production "of the same
title", with music, book, and lyrics by O'Brien.

 The production is a parody tribute to the science fiction and horror B movies of the 1930s through to the early 1970s.

 Along with O'Brien, the film stars Tim Curry, Susan Sarandon, and Barry Bostwick and is narrated by Charles Gray with cast members from the original Royal
Court Theatre, Roxy Theatre, and Belasco Theatre productions.
</item 8>

<item 9>
Obsession (1976 film)
Obsession is a 1976 psychological thriller/mystery film directed by Brian De Palma, starring Cliff Robertson, Geneviève Bujold, John Lithgow, and Stocker
Fontelieu.

 The screenplay was by Paul Schrader, from a story by De Palma and Schrader.

 Bernard Herrmann provided the film's soundtrack prior to his death in 1975.

 The story is about a New Orleans businessman who is haunted by guilt following the death of his wife and daughter during a kidnapping-rescue attempt.

 Years after the tragedy, he meets and falls in love with a young woman who is the exact look-alike of his long dead wife.
</item 9>

<item 10>
Little Fauss and Big Halsy
Little Fauss and Big Halsy is a 1970 film directed by Sidney J. Furie, starring Robert Redford and Michael J. Pollard, also featuring Lauren Hutton, Noah Beery,
Jr. and Lucille Benson.
</item 10>
=== END ITEM id=00211e070a78eca0 ===

=== ITEM id=da6fbdd4c9b580f5 ===
Gold sentences and facts:
Sentence 0:  Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any affiliation with the work.
  Fact 0.0: The film takes its title from an album by Leonard Cohen.
  Fact 0.1: Ira Nadel wrote a biography of Leonard Cohen.
  Fact 0.2: The subject of the film is not Leonard Cohen.
  Fact 0.3: The subject of the film has no affiliation with the work.
Sentence 1: Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and painter.
  Fact 1.0: Leonard Norman Cohen was born on September 21, 1934.
  Fact 1.1: Leonard Norman Cohen died on November 7, 2016.
  Fact 1.2: Leonard Norman Cohen was a Canadian singer.
  Fact 1.3: Leonard Norman Cohen was a Canadian songwriter.
  Fact 1.4: Leonard Norman Cohen was a Canadian musician.
  Fact 1.5: Leonard Norman Cohen was a Canadian poet.
  Fact 1.6: Leonard Norman Cohen was a Canadian novelist.
  Fact 1.7: Leonard Norman Cohen was a Canadian painter.

Retrieved items:
<item 1>
The Cold Nose
The Cold Nose is the debut album by Department of Eagles.

 It has been reissued twice.

 It was originally released in the US as The Whitey On The Moon UK LP on Isota Records on October 21, 2003, and reissued in the UK as The Cold Nose on Melodic
Records on January 8, 2008.

 The American Dust October 7, 2008 US reissue contains four bonus tracks.

 The album is sample-heavy and multilayered.

 The second track, "Sailing By Night," contains a sample of the song "Meetings Along The Edge" from the Philip Glass and Ravi Shankar project "Passages", as
well as a sample from the Yellow Magic Orchestra track, Rydeen.

 The third track "Noam Chomsky Spring Break 2002" samples the Regina Spektor song "Prisoners".

 It also samples Astor Piazzolla's composition "Oblivion" in the opening and closing sequences.
</item 1>

<item 2>
For Those Who Think Young (album)
For Those Who Think Young, appearing on the album cover as (for those who think young) and originally to be entitled for those who think jung, was the third
album by Canadian new wave band Rough Trade; it was released in 1981 (True North TN-48 in Canada; Boardwalk NB-33261-1, US; Big Time, UK; CBS 85385, The
Netherlands).

 It climbed to #9 in Canada on the "RPM" Top 50 Albums Chart on November 7, 1981 (putting it at #1 on the CANCON Chart listing), and held the position for three
weeks, dropping out of the Top 50 after sixteen weeks on February 6 of the following year.

 It was certified gold in Canada by the CRIA on November 1, 1981.

 The single "All Touch" gave the band its biggest commercial success, reaching #12 in Canada on the "RPM" Top 50 Singles Chart (#2 on the CANCON Chart) and #58
on the U.S. ""Billboard" Hot 100".
</item 2>

<item 3>
Various Positions Tour
The Various Positions Tour was a concert tour by Leonard Cohen, in support of his album "Various Positions", released in 1984 (Canada) and 1985 in Europe and
the United States.
</item 3>

<item 4>
Various Positions (film)
Various Positions is a 2002 film by Vancouver, BC, lawyer and filmmaker Ori Kowarsky, starring Carly Pope and Tygh Runyan.

 "Various Positions" won the 2002 Prix de Montréal at the Montreal World Film Festival.

 Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any
affiliation with the work.
</item 4>

<item 5>
Garcia Plays Dylan
Garcia Plays Dylan is an album composed of various live performances featuring Jerry Garcia playing covers of Bob Dylan songs.

 It is culled from performances from 1973–1995, and features Garcia playing with Grateful Dead, Legion of Mary, Jerry Garcia Band, and Garcia-Saunders.

 Garcia takes lead vocals on all tracks.
</item 5>

<item 6>
Leonard Cohen
Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and
painter.

 His work explored religion, politics, isolation, sexuality, and personal relationships.

 Cohen was inducted into the Canadian Music Hall of Fame, the Canadian Songwriters Hall of Fame, and the Rock and Roll Hall of Fame.

 He was a Companion of the Order of Canada, the nation's highest civilian honor.

 In 2011, Cohen received one of the Prince of Asturias Awards for literature and the ninth Glenn Gould Prize.
</item 6>

<item 7>
Jack Johnson (album)
Jack Johnson, later reissued as A Tribute to Jack Johnson, is a 1971 studio album and soundtrack by American jazz trumpeter, composer, and bandleader Miles
Davis.

 In 1970, Davis was asked by Bill Cayton to record music for his documentary of the same name on the life of boxer Jack Johnson.

 Johnson's saga resonated personally with Davis, who wrote in the album's liner notes of Johnson's mastery as a boxer, his affinity for fast cars, jazz,
clothes, and beautiful women, his unreconstructed blackness, and his threatening image to white men.

 This was the second film score he had composed, after "Ascenseur pour l'échafaud" in 1957.
</item 7>

<item 8>
Love Can't Wait (Nick Carter song)
"Love Can't Wait" is a song by American singer-songwriter Nick Carter, released as the lead Canadian single from his second studio album, "I'm Taking Off".

 The single was released on June 21, 2011.

 The song only appears on the Canadian version of the album, and has not been released in any other territories.

 The music video for the song features Carter performing the song with the backdrop of a motorway system.

 The music video also featured Canadian pop star Shawn Desman, who co-wrote the song.

 The single peaked at #2 on the Canadian Singles Chart, based entirely on download sales.

 Carter ran a competition for amateur mixers to remix the track, with the winning remixer earning $1500 and their remix appearing on Carter's remix album, "I'm
Taking Off: Relaunched and Remixed".
</item 8>

<item 9>
Of Montreal
of Montreal is an American experimental pop band from Athens, Georgia.

 It was founded by frontman Kevin Barnes in 1996, named after a failed romance between Barnes and a woman "of Montreal."

 The band is identified as part of the Elephant 6 collective.

 Throughout its existence, of Montreal's musical style has evolved considerably and drawn inspiration from numerous 1960s psychedelic pop acts.
</item 9>

<item 10>
The Promised Land (Sagol 59 album)
The Promised Land: The Grateful Dead / Jerry Garcia Hebrew Project is a 2015 album by Sagol 59 and Ami Yares, featuring Hebrew language covers of songs by The
Grateful Dead and Jerry Garcia.

 It was released on April 17, 2015.

 The album was followed, in August 2015, by "The Promised Land Acoustic Sessions" EP, which contained four additional recordings: Hebrew renditions of "Ripple",
"Brown Eyed Women", "Friend of The Devil" and "Ship Of Fools".

 "The Pita Tapes" followed in March 2016, containing two unreleased demo recordings of songs often covered by The Grateful Dead and Garcia (Merle Haggard's
"Mama Tried" and Johnny Russell's "Catfish John").

 A nine-track limited edition Vinyl LP was released on April 2016.

 On August 1, 2016, studio recordings of "Deep Elem Blues" and "Half-Step Mississippi Uptown Toodeloo" were released digitally, to coincide with Garcia's
birthday.
</item 10>
=== END ITEM id=da6fbdd4c9b580f5 ===

=== ITEM id=d8763226765f22ec ===
Gold sentences and facts:
Sentence 0:  The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part, which he played on a Gibson acoustic.
  Fact 0.0: The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part.
  Fact 0.1: Sammy Hagar played the rhythm guitar part on a Gibson acoustic.
Sentence 1: OU812 (pronounced "Oh You Ate One Too") is the eighth studio album by American hard rock band Van Halen, released in 1988, and the second to feature vocalist Sammy Hagar.
  Fact 1.0: OU812 is pronounced 'Oh You Ate One Too'.
  Fact 1.1: OU812 is the eighth studio album by American hard rock band Van Halen.
  Fact 1.2: OU812 was released in 1988.
  Fact 1.3: OU812 is the second Van Halen album to feature vocalist Sammy Hagar.

Retrieved items:
<item 1>
Keiko Utoku
Keiko Utoku (宇徳敬子 , "Utoku Keiko" , born April 7, 1967 in Izumi, Kagoshima, Japan) is a Japanese singer and songwriter under UK Sweet label.Keiko made first
debut with chorus part in band B.B.Queens then joined with J-pop female band Mi-Ke from 1991 to 1993, before starting a solo career.

 As a child, Utoku listened to a lot of music, including music made by Momoe Yamaguchi.

 She has written and sung many songs, such as "Wasurenagusa", and Sonic the Hedgehog fans remember her for her vocal work in the original version of the Sonic
CD video game's opening and closing themes in Japan and Europe, "Sonic – You Can Do Anything" and "Cosmic Eternity – Believe In Yourself" in 1993.

 In addition, she also provided Sonic's vocals in the game.

 At recent dates, she has released 15 singles, 5 digital singles, 4 studio albums, 1 mini and 2 best albums.
</item 1>

<item 2>
OU812
OU812 (pronounced "Oh You Ate One Too") is the eighth studio album by American hard rock band Van Halen, released in 1988, and the second to feature vocalist
Sammy Hagar.

 Van Halen started work on the album in September 1987 and completed it in April 1988, just one month before its release.
</item 2>

<item 3>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 3>

<item 4>
&quot;Happy&quot; in Galoshes
"Happy" in Galoshes is the second solo album by American rock singer Scott Weiland.

 Weiland, known for his roles as the lead singer in Stone Temple Pilots and Velvet Revolver, released his first album, "12 Bar Blues" in 1998.

 Ten years later, ""Happy" in Galoshes" served as the official follow-up.

 Two versions have been released, a single disc and double disc deluxe version—the deluxe edition offers a second disc of 10 extra tracks.

 The album was released November 25, 2008 on Weiland's own Softdrive Records.

 Produced by Doug Grean and Scott Weiland, with select tracks recorded by Steve Albini, the album features guest appearances by Paul Oakenfold and No Doubt
members Adrian Young, Tony Kanal, and Tom Dumont.

 Oakenfold appears on a cover of David Bowie's "Fame."

 Weiland has cited Bowie as one of his main influences.
</item 4>

<item 5>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 5>

<item 6>
OU812 Tour
The OU812 Tour was a concert tour by hard rock band Van Halen.

 It was the second tour to include dates in Japan, and was the second with Sammy Hagar as vocalist (who had settled into the role more by this point).

 It followed the Monsters of Rock Tour 1988, which had formed the first part of the promotion for the "OU812" album.
</item 6>

<item 7>
Keisuke Kuwata (album)
Keisuke Kuwata is the eponymous studio album recorded by Keisuke Kuwata, a frontman of the Japanese rock band Southern All Stars.

 It was released by Taishita label under the Victor Entertainment in July 1988, shortly after the 10th anniversary of the band.

 Aside from the 1982 live recording album "Kamon Yuzo and Victor Wheels Live" released under the pseudonym Yūzō Kamon, it was first Kuwata's solo effort.
</item 7>

<item 8>
Hoarse &amp; Roaring
Hoarse & Roaring is the debut full-length album by neofolk band Parlor Hawk, released in 2010 on Northplatte Records.

 The album was produced by Joshua James and featured Neon Trees bassist Branden Campbell on each track (except "Saddest Song") as well as Fictionist member
Stuart Maxfield on the tracks "14 Years" and "Home".

 It was mixed by Todd Burke (Ben Harper, Jack Johnson) and mastered by Reuben Cohen (Edward Sharpe & The Magnetic Zeros, Bruno Mars).
</item 8>

<item 9>
Finish What Ya Started
"Finish What Ya Started" is a song by Van Halen taken from their 1988 album "OU812".

 Despite the album being seemingly complete, Eddie Van Halen came up with the riff at 2 in the morning and went down to his then-neighbor Sammy Hagar to show
it.

 Hagar let Eddie in, and the two played guitars in his balcony until they had a completed song.

 Once Eddie left, Hagar decided to write the lyrics despite being late at night.

 The theme wound up being unfulfilled sex, summed up by Hagar as "blue balls.

 In the song, Eddie Van Halen recorded his guitar part on a Fender Stratocaster plugged direct into the studio mixing console.

 The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part, which he played on a Gibson acoustic.
</item 9>

<item 10>
Joe Gooch
Joe Gooch (born 3 May 1977) was Ten Years After's most recent lead vocalist and lead guitarist.
</item 10>
=== END ITEM id=d8763226765f22ec ===

=== ITEM id=75f18db0d77ea754 ===
Gold sentences and facts:
Sentence 0: German Spitz is used to refer to both a breed of dog and category or type of dog.
  Fact 0.0: German Spitz is used to refer to a breed of dog.
  Fact 0.1: German Spitz is used to refer to a category or type of dog.
Sentence 1:  Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.
  Fact 1.0: Several modern breeds have been developed from the German Spitz.
  Fact 1.1: These modern breeds are either registered as separate breeds.
  Fact 1.2: These modern breeds are either registered as varieties of German Spitz.
Sentence 2:  The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that would later become known as the English Springer Spaniel.
  Fact 2.0: The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels.
  Fact 2.1: Attempts were made to use the term to specify a breed that would later become known as the English Springer Spaniel.

Retrieved items:
<item 1>
Welsh Springer Spaniel
The Welsh Springer Spaniel is a breed of dog and a member of the spaniel family.

 Thought to be comparable to the old Land Spaniel, they are similar to the English Springer Spaniel and historically have been referred to as both the Welsh
Spaniel and the Welsh Cocker Spaniel.

 They were relatively unknown until a succession of victories in dog trials by the breed increased its popularity.

 Following recognition by The Kennel Club in 1902, the breed gained the modern name of Welsh Springer Spaniel.

 The breed's coat only comes in a single colour combination of white with red markings, usually in a piebald pattern.

 Loyal and affectionate, they can become very attached to family members and are wary of strangers.

 Health conditions are limited to those common among many breeds of dog, although they are affected more than average by hip dysplasia and some eye conditions.

 They are a working dog, bred for hunting, and while not as rare as some varieties of spaniel, they are rarer than the more widely known English Springer
Spaniel with which they are sometimes confused.
</item 1>

<item 2>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 2>

<item 3>
American Cocker Spaniel
The American Cocker Spaniel is a breed of sporting dog.

 It is a spaniel type dog that is closely related to the English Cocker Spaniel; the two breeds diverged during the 20th century due to differing breed
standards in America and the UK.

 In the United States, the breed is usually called the Cocker Spaniel, while elsewhere in the world, it is called the American Cocker Spaniel in order to
differentiate between it and its English cousin, which was already known as "Cocker Spaniel" before the American variety was created.

 The word "cocker" is commonly held to stem from their use to hunt woodcock in England, while "spaniel" is thought to be derived from the type's origins in
Spain.
</item 3>

<item 4>
King Charles Spaniel
The King Charles Spaniel (also known as the English Toy Spaniel) is a small dog breed of the spaniel type.

 In 1903, the Kennel Club combined four separate toy spaniel breeds under this single title.

 The other varieties merged into this breed were the Blenheim, Ruby and Prince Charles Spaniels, each of which contributed one of the four colours available in
the breed.
</item 4>

<item 5>
Picardy Spaniel
The Picardy Spaniel is a breed of dog developed in France for use as a gundog.

 It is related to the Blue Picardy Spaniel, and still has many similarities, but the Picardy Spaniel is the older of the two breeds.

 It is thought to be one of the two oldest continental spaniel breeds and was favoured by the French nobility, remaining popular for hunting after the French
Revolution due to its weather resistant coat that enabled it to hunt in a variety of conditions and terrain.

 However its popularity waned following the influx of English hunting breeds in the early 20th century.

 Slightly smaller than an English Setter but larger than most of its spaniel cousins, it has no major health issues although as with many breeds with pendulous
ears, it can be prone to ear infections.
</item 5>

<item 6>
German Spitz
German Spitz is used to refer to both a breed of dog and category or type of dog.

 Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.

 All the "German Spitz type" dogs are dogs of the Spitz type of German origin.

 The Großspitz, Mittelspitz, and Kleinspitz breeds of "German Spitz type" are also called the German Spitz in English.
</item 6>

<item 7>
American Eskimo Dog
The American Eskimo Dog is a breed of companion dog originating in Germany.

 The American Eskimo is a member of the Spitz family.

 The breed's progenitors were German Spitz, but due to anti-German prejudice during the First World War, it was renamed "American Eskimo Dog".

 Although modern American Eskimos have been exported as German Spitz Gross (or Mittel, depending on the dog's height), the breeds have diverged and the
standards are significantly different.

 In addition to serving as a watchdog and companion, the American Eskimo Dog also achieved a high degree of popularity in the United States in the 1930s and
1940s as a circus performer.
</item 7>

<item 8>
German Spitz (Klein)
The German Spitz Klein is a breed of dog of the German Spitz type.

 They are usually classed as a toy or utility breed.
</item 8>

<item 9>
Norfolk Spaniel
The Norfolk Spaniel or Shropshire Spaniel is an extinct breed of dog since the early 20th century.

 It was originally thought to have originated from the work of one of the Dukes of Norfolk, but this theory was disproven after being in doubt during the later
part of the 19th century.

 The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that
would later become known as the English Springer Spaniel.
</item 9>

<item 10>
English Springer Spaniel
The English Springer Spaniel is a breed of gun dog in the Spaniel family traditionally used for flushing and retrieving game.

 It is an affectionate, excitable breed with a typical lifespan of twelve to fourteen years.

 They are very similar to the Welsh Springer Spaniel and are descended from the Norfolk or Shropshire Spaniels of the mid-19th century; the breed has diverged
into separate show and working lines.

 The breed suffers from average health complaints.

 The show-bred version of the breed has been linked to "rage syndrome", although the disorder is very rare.

 It is closely related to the Welsh Springer Spaniel and very closely to the English Cocker Spaniel; less than a century ago, springers and cockers would come
from the same litter.

 The smaller "cockers" hunted woodcock while the larger littermates were used to flush, or "spring," game.

 In 1902, The Kennel Club recognized the English Springer Spaniel as a distinct breed.

 They are used as sniffer dogs on a widespread basis.

 The term "Springer" comes from the historic hunting role, where the dog would flush (spring) birds into the air.
</item 10>
=== END ITEM id=75f18db0d77ea754 ===
