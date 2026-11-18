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

This file has 10 items.

=== ITEM id=fb3f2572a636b1c5 ===
Gold sentences and facts:
Sentence 0: Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.
  Fact 0.0: Underdog is a 2007 American film.
  Fact 0.1: Underdog is a family superhero comedy film.
  Fact 0.2: Underdog is based on the 1960s cartoon series of the same name.
Sentence 1: Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi, a Life in the Woods" by Austrian author Felix Salten.
  Fact 1.0: Bambi is a 1942 American animated film.
  Fact 1.1: Bambi was directed by David Hand.
  Fact 1.2: David Hand supervised a team of sequence directors on Bambi.
  Fact 1.3: Bambi was produced by Walt Disney.
  Fact 1.4: Bambi is based on the book 'Bambi, a Life in the Woods'.
  Fact 1.5: The book 'Bambi, a Life in the Woods' was written by Austrian author Felix Salten.

Retrieved items:
<item 1>
Underdog (film)
Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.

 Directed by Frederik Du Chau and written by Joe Piscatella, Adam Rifkin, and Craig A. Williams, the film stars Jim Belushi, Peter Dinklage, John Slattery, and
Patrick Warburton with the voice talents of Jason Lee, Amy Adams, and Brad Garrett.

 Unlike the TV series, the Underdog character is portrayed as a regular dog rather than an anthropomorphic one.

 Underdog, voiced by Jason Lee, was played by a lemon beagle named Leo sporting a red sweater and a blue cape.

 The film grossed $65.3 million worldwide.
</item 1>

<item 2>
Pete (Disney)
Pete (also called Peg-Leg Pete, Pistol Pete and Black Pete, among other names) is an anthropomorphic cartoon character created in 1925 by Walt Disney and Ub
Iwerks.

 He is a character of The Walt Disney Company and often appears as a nemesis and the main antagonist in Mickey Mouse universe stories.

 He was originally an anthropomorphic bear but with the advent of Mickey Mouse in 1928, he was defined as a cat.

 Pete is the oldest continuing Disney character, having debuted three years before Mickey Mouse in the cartoon "Alice Solves the Puzzle" (1925).
</item 2>

<item 3>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 3>

<item 4>
Bambi
Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi,
a Life in the Woods" by Austrian author Felix Salten.

 The film was released by RKO Radio Pictures on August 13, 1942, and is the fifth Disney animated feature film.
</item 4>

<item 5>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 5>

<item 6>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 6>

<item 7>
Bosko the Musketeer
Bosko the Musketeer is an American animated short film.

 It is a "Looney Tunes" cartoon, featuring Bosko, the first star of the series.

 It was released on August 12, 1933, although some sources note September 16 as a date; this is problematic, as that would imply that the last films featuring
Bosko as the star of Warner Bros. cartoons were released after the first film featuring Buddy, the second star of the series.

 It was, like most "Looney Tunes" of the time, directed by Hugh Harman; Frank Marsales was the musical director.
</item 7>

<item 8>
Get a Horse!
Get a Horse!

 is a 2013 American 3D animated slapstick comedy short film, produced by Walt Disney Animation Studios.

 Combining black-and-white hand-drawn animation and color CGI animation, the short features the characters of the late 1920s "Mickey Mouse" cartoons, and
features archival recordings of Walt Disney in his posthumous role as Mickey Mouse.

 It is the first original Mickey Mouse theatrical animated short since "Runaway Brain" (1995), and the first appearance of Oswald the Lucky Rabbit in a Disney
animated production in 85 years.
</item 8>

<item 9>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 9>

<item 10>
Pinocchio (1940 film)
Pinocchio is a 1940 American animated musical fantasy film produced by Walt Disney Productions and based on the Italian children's novel "The Adventures of
Pinocchio" by Carlo Collodi.

 It was the second animated feature film produced by Disney, made after the success of "Snow White and the Seven Dwarfs" (1937).
</item 10>
=== END ITEM id=fb3f2572a636b1c5 ===

=== ITEM id=e585bea61df2fcd2 ===
Gold sentences and facts:
Sentence 0: Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.
  Fact 0.0: Adam Fox lived from 1883 to 1977.
  Fact 0.1: Adam Fox was a Canon.
  Fact 0.2: Adam Fox was the Dean of Divinity at Magdalen College, Oxford.
Sentence 1:  He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.
  Fact 1.0: Adam Fox was one of the first members of the Inklings.
  Fact 1.1: The Inklings was a literary group.
  Fact 1.2: The Inklings also included C.S. Lewis.
  Fact 1.3: The Inklings also included J.R.R. Tolkien.
Sentence 2: The Inklings was an informal literary discussion group associated with the University of Oxford, England, for nearly two decades between the early 1930s and late 1949.
  Fact 2.0: The Inklings was an informal literary discussion group.
  Fact 2.1: The Inklings was associated with the University of Oxford, England.
  Fact 2.2: The Inklings lasted for nearly two decades.
  Fact 2.3: The Inklings operated between the early 1930s and late 1949.

Retrieved items:
<item 1>
James F. Light
James F. Light was an American literary scholar, university vice president, and provost.

 During his academic career, he helped revive the works of satirist Nathanael West, with the first book length critical study of his work, "Nathanael West: An
Interpretive Study", (Northwestern Univ.

 Press, 1961).

 He was also the leading authority on John William De Forest, the early American realist whose work he critiqued in "John William De Forest" (Twayne Pub.,
1965), and he wrote extensively on J. D. Salinger, Robert Penn Warren and others.
</item 1>

<item 2>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 2>

<item 3>
Andrew Benjamin
Andrew Benjamin (born 1952, Australia) is an Australian philosopher.

 Currently he holds two academic posts: Distinguished Professor of Philosophy and the Humanities at Kingston University, London and Professor of Philosophy and
Jewish Thought in the Department of Philosophy and the Australian Centre for Jewish Civilization at Monash University, Melbourne, Australia.

 Benjamin first came to critical attention with his writings in continental philosophy, writing articles and editing books on the thinking of Martin Heidegger,
Theodor Adorno, Walter Benjamin, Julia Kristeva and Jean-François Lyotard.

 Benjamin has become involved in the field of architecture, to the extent that he has also taught in various schools of architecture in UK, US and Australia.
</item 3>

<item 4>
Falmouth University
Falmouth University (Cornish: Pennskol Aberfala ) is a specialist University for the creative industries based in Falmouth and Penryn, Cornwall, England.

 Founded in 1902, it had previously been the Falmouth School of Art, Falmouth College of Art and Design and then Falmouth College of Arts until it received
taught degree-awarding powers, and the right to use the title "University College", in March 2005.

 In April 2008, University College Falmouth merged with Dartington College of Arts, adding a range of Performance courses to its portfolio.

 On 27 November 2012, a communication was released to the staff and students and local press that 'University College Falmouth is to be granted full university
status in a move that will further its ambition to become one of the top five arts universities in the world.'

 On 9 December 2012, the University College was officially granted full university status by the Privy Council.

 Falmouth University was judged by The Sunday Times to be the UK’s top arts university from 2015-2017.
</item 4>

<item 5>
Alpha Gamma
Alpha Gamma (ΑΓ) was an American collegiate fraternity.

 It was founded at Cumberland University in Lebanon, Tennessee in 1867.

 About 21 chapters developed, with prominent chapters at Washington & Jefferson College, Trinity University, Mercersburg College, Southwestern Presbyterian
University, Cumberland University, and the West Virginia University.

 The fraternity's badge consisted of a golden shield with a globe, six stars, and a pennant with "ΑΓ."
</item 5>

<item 6>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 6>

<item 7>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 7>

<item 8>
Inklings
The Inklings was an informal literary discussion group associated with the University of Oxford, England, for nearly two decades between the early 1930s and
late 1949.

 The Inklings were literary enthusiasts who praised the value of narrative in fiction and encouraged the writing of fantasy.
</item 8>

<item 9>
Eagle Street College
The Eagle Street College was an informal literary society established in 1885 at the home of James William Wallace in Eagle Street, Bolton, to read and discuss
literary works, particularly the poetry of Walt Whitman, (1819–91).

 The group subsequently became known as the Bolton Whitman Fellowship or Whitmanites.

 Its founder members were Wallace, Dr John Johnston and Fred Wild.

 The group held an annual 'Whitman Day' celebration around the 31 May which was the poet's birthday.
</item 9>

<item 10>
Litbits
Litbits is a literary discussion radio programme and podcast, hosted by academic Adam Smyth and literary journalist James Kidd.

 Guests on the programme have included Polly Russell, Paul Myerscough, Alex von Tunzelmann, Samantha Ellis, Jonathan Beckman, Hallie Rubenhold, Steve Rose, Joe
Brooker, Dennis Duncan, Dinah Roe, Giles Milton and Liane Strauss.

 Subjects covered include Rooms and Procrastination.
</item 10>
=== END ITEM id=e585bea61df2fcd2 ===

=== ITEM id=c2e4bbaf103af132 ===
Gold sentences and facts:
Sentence 0: The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
  Fact 0.0: The International Grenfell Association (IGA) is an organization.
  Fact 0.1: The International Grenfell Association was founded by Sir Wilfred Grenfell.
  Fact 0.2: The International Grenfell Association provides health care.
  Fact 0.3: The International Grenfell Association provides education.
  Fact 0.4: The International Grenfell Association provides religious services.
  Fact 0.5: The International Grenfell Association provides rehabilitation.
  Fact 0.6: The International Grenfell Association provides other social activities.
  Fact 0.7: The International Grenfell Association serves fisherman and coastal communities in northern Newfoundland.
  Fact 0.8: The International Grenfell Association serves fisherman and coastal communities on the coast of Labrador.
Sentence 1: Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
  Fact 1.0: Sir Wilfred Thomason Grenfell held the title KCMG.
  Fact 1.1: Sir Wilfred Thomason Grenfell was born on 28 February 1865.
  Fact 1.2: Sir Wilfred Thomason Grenfell died on 9 October 1940.
  Fact 1.3: Sir Wilfred Thomason Grenfell was a medical missionary to Newfoundland.
  Fact 1.4: Sir Wilfred Thomason Grenfell was a medical missionary to Labrador.

Retrieved items:
<item 1>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 1>

<item 2>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 2>

<item 3>
Training centre for release of the Atma-energy
Training centre for release of the Atma-energy (German: "Trainingszentrum zur Freisetzung der Atmaenergie"), also known as "Atman Foundation", was a new
religious movement active mainly on the island of Tenerife and in Germany.

 This sect was originally a splinter group of the Brahma Kumaris and is known for a police and media scare in which an alleged attempt to commit ritual suicide
took place in Teide National Park in Tenerife.

 The group believed in the end of the world but according to the religious studies scholar Georg Schmid and the sociologist Massimo Introvigne had no intention
of collective suicide.
</item 3>

<item 4>
Ellis Gowing
Gowing was educated at Fort Street High School and the University of Sydney; and ordained in 1907.

 He began his career as a Curate at The Oaks, New South Wales, after which he served at Armidale Cathedral.

 Coming to England he was on the staff of St James the Less, Bethnal Green then domestic chaplain to the Bishop of Chelmsford from 1914 to 1917.

 He was vicar of Prittlewell from 1917 until his death; an honorary canon of Chelmsford Cathedral from 1921 to 1938 and Rural Dean of Canewdon from 1918 to
1938.

 He was given the Freedom of the County Borough of Southend-on-Sea in 1953.
</item 4>

<item 5>
Wilfred Grenfell
Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
</item 5>

<item 6>
The text surveys diverse religious, political, civic, and governmental groups. The Atman Foundation, a Brahma Kumaris offshoot, faced a false collective-suicide
scare in Tenerife. Montenegro’s 2006 “Eagle’s Flight” operation arrested 17 alleged militants. Greece’s anarchist Conspiracy of Fire Cells carried out bombings;
the Indian National Council supported independence efforts. Other entries cover Sahara-Sahel environmental work, a declining Indian Trotskyist group,
Arab-American civil rights, diplomacy,
</item 6>

<item 7>
Andrew Bryan (Baptist)
Andrew Bryan (1737–1812) founded First Bryan Baptist Church, affectionately called the Mother Church of Black Baptists, and First African Baptist Church of
Savannah in Savannah, Georgia, the first black Baptist churches to be established in America.

 Bryan was the former slave of Jonathan Bryan.
</item 7>

<item 8>
Angelino Apelar
Angelino "Andy" Baldemor Apelar (June 16, 1927 - January 23, 2006) was an Evangelical Christian leader in the Filipino American community, and a founder of the
Association of Filipino Churches (AFC) of the Christian and Missionary Alliance.

 He had served as a missionary/pastor and radio broadcaster in the Philippines before moving to the United States to start the AFC.

 He served as president for several terms.

 He graduated from the Far Eastern Bible Institute and Seminary, where he also met his future wife, Purita Palomar, a member of one of Capiz province's
influential families.

 They have two children: Faith and Jemuel Apelar.
</item 8>

<item 9>
International Grenfell Association
The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and
rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
</item 9>

<item 10>
The entries cover British and Australian church heritage. London’s St Paul’s, founded on the site of a church dating to 604, is Wren’s Grade I-listed Baroque
cathedral, rebuilt after the Great Fire. Rockhampton’s St Paul’s Cathedral (1883–1953) and adjacent hall (built circa 1900) are Queensland-listed. Augustine
became Canterbury’s first archbishop in 597. Other cathedrals include medieval Lichfield (England’s only
</item 10>
=== END ITEM id=c2e4bbaf103af132 ===

=== ITEM id=aeebf8a5eed914c2 ===
Gold sentences and facts:
Sentence 0: Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.
  Fact 0.0: Emmitt H. Ford was born on December 13, 1943.
  Fact 0.1: Emmitt H. Ford died on November 10, 2014.
  Fact 0.2: Emmitt H. Ford was an American politician.
  Fact 0.3: Emmitt H. Ford was from Tennessee.
Sentence 1:  He was also an uncle of former United States Congressman Harold Ford Jr.
  Fact 1.0: Emmitt H. Ford was an uncle of Harold Ford Jr.
  Fact 1.1: Harold Ford Jr. is a former United States Congressman.
Sentence 2:  He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.
  Fact 2.0: Emmitt H. Ford is the son of Harold Ford Sr.
  Fact 2.1: Harold Ford Sr. is a former Congressman.
  Fact 2.2: Harold Ford Sr. held the same seat for 22 years.

Retrieved items:
<item 1>
Emmitt Ford
Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.

 He represented the 86th district encompassing Shelby County from 1975–1981.

 He was also an uncle of former United States Congressman Harold Ford Jr.
</item 1>

<item 2>
Doug Ford Jr.
Douglas Bruce Ford Jr. (born November 20, 1964) is a Canadian businessman and politician in Toronto, Ontario.

 Ford was Toronto City Councillor for Ward 2 Etobicoke North in Toronto from 2010 to 2014 at the same time that his brother, Rob Ford (1969–2016), was mayor of
Toronto.

 Their father, Doug Ford Sr., was a Member of Provincial Parliament from 1995 to 1999 and founded Deco Labels and Tags, a printing business operating in Canada
and the United States.

 The business is now owned by Doug Ford Jr., who has served as company president since 2002, and his brother, Randy.
</item 2>

<item 3>
United States Senate election in Tennessee, 2006
The 2006 United States Senate election in Tennessee was held on November 7, 2006.

 The election winner, Bob Corker, served his term between January 3, 2007 and January 3, 2013.

 Corker replaced Republican Bill Frist in the Senate, who retired upon the end of his second term in 2007.

 Corker was the Republican nominee, and the Democratic nominee was Harold Ford, Jr..

 The race between Ford and Corker was one of the most competitive Senate races of 2006, with Corker winning the race by less than three percent of the vote.

 Corker was the only non-incumbent Republican to win a U.S. Senate seat in 2006.

 Since 1994, the Republican Party has held both of Tennessee's Senate seats.
</item 3>

<item 4>
John Y. Brown Jr.
John Young Brown Jr. (born December 28, 1933) is an American politician, entrepreneur, and businessman from the U.S. state of Kentucky.

 He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a
multimillion-dollar restaurant chain.
</item 4>

<item 5>
John Ford (Tennessee politician)
John N. Ford (born May 3, 1942 in Memphis, Tennessee), is a former Democratic member of the Tennessee State Senate and a member of Tennessee's most prominent
African-American political family.

 He is the older brother of former U.S. Representative Harold Ford, Sr. and the uncle of former Tennessee U.S. Representative and 2006 United States Senate
candidate Harold Ford, Jr. He is a member of the Ford political family from Memphis.
</item 5>

<item 6>
John Anderson (Maine politician)
John Anderson (1792–1853) was a Maine politician.

 Anderson served as United States Representative from Maine from 1825-1833.
</item 6>

<item 7>
L. R. Ford Jr.
Lester Randolph Ford Jr. (born September 23, 1927 – February 26, 2017) was an American mathematician specializing in network flow problems.

 He was the son of mathematician Lester R. Ford Sr.
</item 7>

<item 8>
Ophelia Ford
Ophelia Ford (born July 5, 1950) was a member of the Tennessee Senate.

 She is the younger sister of former state senator John Ford and former Congressman Harold Ford, Sr., and the aunt of former Congressman Harold Ford, Jr. She
represented Senate District 29, which covers South Memphis and North Memphis.
</item 8>

<item 9>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 9>

<item 10>
Harold Ford Jr.
Harold Eugene Ford Jr. (born May 11, 1970) is an American politician and author who served five terms in the United States House of Representatives as a member
of the Democratic Party from  's 9 congressional district , centered in Memphis, from 1997 to 2007.

 He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.

 In 2006, Ford made an unsuccessful bid for the US Senate seat vacated by retiring Bill Frist.

 He is a member of the Ford political family from Memphis.

 Ford was the last chairman of the Democratic Leadership Council (DLC).
</item 10>
=== END ITEM id=aeebf8a5eed914c2 ===

=== ITEM id=971055e489d2d5f3 ===
Gold sentences and facts:
Sentence 0: St. Elizabeths Hospital opened in 1855 as the first federally operated psychiatric hospital in the United States.
  Fact 0.0: St. Elizabeths Hospital opened in 1855.
  Fact 0.1: St. Elizabeths Hospital opened as the first federally operated psychiatric hospital in the United States.
Sentence 1:  Founded in 1861, it is the longest continuously operating hospital in the District.
  Fact 1.0: St. Elizabeths Hospital was founded in 1861.
  Fact 1.1: St. Elizabeths Hospital is the longest continuously operating hospital in the District.

Retrieved items:
<item 1>
Vassar Brothers Medical Center
Vassar Brother Medical Center (locally known as Vassar Hospital or VBMC) is a major medical facility located in the city of Poughkeepsie, New York that is a
member of the Health Quest network, a nonprofit family of hospitals and healthcare centers in the Hudson Valley area.

 VBMC is one of two major medical centers located in Dutchess County, New York, the other being Mid-Hudson Regional Hospital of Westchester Medical Center.

 The hospital was first incorporated in 1882 under the name Vassar Brothers Hospital, following contributions made by John Guy Vassar following the death of his
brother.

 The hospital opened its doors in 1887, and initially had 40 beds divided up into four wards, each containing ten beds.

 At the time the hospital opened it also contained a labor and delivery ward, a children's ward, a nursery, three private rooms, and two isolation rooms.

 Several major renovations occurred in 1983, 1991, and 2001; adding critical care areas, a new operating and delivery wing, and a cancer care center,
respectively.
</item 1>

<item 2>
Graph query row:
hospital: Providence Hospital
date: 1861
relationship: None
</item 2>

<item 3>
Montefiore New Rochelle Hospital
Montefiore New Rochelle Hospital (formerly Sound Shore Medical Center) is a community-based, teaching hospital located at 16 Guion Place in the city of New
Rochelle, in Westchester County, New York, and affiliated with the Albert Einstein College of Medicine.

 The hospital opened in 1892 as New Rochelle Hospital.

 On November 6, 2013, Sound Shore was acquired by the Bronx-based Montefiore Medical Center, the University Hospital for the Albert Einstein College of
Medicine, was renamed Montefiore New Rochelle Hospital, and became part of the Montefiore Health System.
</item 3>

<item 4>
St. Elizabeths Hospital
St. Elizabeths Hospital opened in 1855 as the first federally operated psychiatric hospital in the United States.

 Housing over 8,000 patients at its peak in the 1950s, the hospital at one point had a fully functioning medical-surgical unit, a school of nursing, and
accredited internships and psychiatric residencies.

 Its campus was designated a National Historic Landmark in 1990.
</item 4>

<item 5>
Graph query row:
hospital: St. Elizabeths Hospital
date: 1855
relationship: None
</item 5>

<item 6>
St. John's Regional Medical Center (California)
St. John's Regional Medical Center is a hospital located in Oxnard, California in the United States, and is operated by Dignity Health, along with its sister
hospital, St. John's Pleasant Valley Hospital in Camarillo, California.

 The hospital was founded in 1912.

 St. John's Regional Medical Center and St. John's Pleasant Valley Hospital offer comprehensive medical services, including 24-hour emergency medical and
surgical services and care, cancer and oncology care and support, cardiovascular care, community outreach and screenings, diagnostic imaging services,
laboratory services, maternity and women's services, neonatal intensive care, palliative care, patient and family education, rehabilitation services, spine and
orthopedic care, weight loss surgery, wound healing and oxygen therapy, and more.

 Together, St. John's Regional Medical Center and St. John's Pleasant Valley Hospital represent the largest acute-care health organization in Ventura County.

 St. John's hospitals serve all of Ventura County and beyond, including the cities of Camarillo, Moorpark, Oxnard, Port Hueneme, Ventura, and Somis.
</item 6>

<item 7>
Heritage Hospital
Heritage Hospital is a hospital located in Tarboro, North Carolina.

 It is a part of the University Health Systems of Eastern Carolina (UHSEC).

 Edgecombe General Hospital opened as a county-owned hospital in 1916.

 It succeeded Pittman Hospital, which opened in 1901.

 In 1959, the Hill-Burton Act helped combine Edgecombe General Hospital, with three other facilities.

 Edgecombe County sold the hospital to Hospital Corporation of America (HCA) in 1982.

 HCA opened a 127-bed facility in 1985, named Heritage Hospital.

 UHSEC bought Heritage Hospital in 1998 from HCA.

 The hospitals focus is as a community hospital.

 The hospital has 101 general and 16 rehabilitation hospital beds.

 It has five Shared Inpatient/Ambulatory Surgery, two Endoscopy, and one C-Section operating rooms.
</item 7>

<item 8>
Providence Hospital (Washington, D.C.)
Providence Hospital is a 408 bed hospital located in the District of Columbia.

 Founded in 1861, it is the longest continuously operating hospital in the District.

 Providence Hospital is a member of Ascension Health, the largest non-profit health care organization in the United States.
</item 8>

<item 9>
Vassar Brothers Medical Center
In 2002 Vassar Brothers Hospital officially became Vassar Brothers Medical Center.

 In September 2016, VBMC announced a new $500 million expansion which would almost double the number of available rooms.

 Construction on the expansion is expected to be completed in 2019.
</item 9>

<item 10>
History of psychiatric hospitals in Pittsburgh
Pittsburgh, Pennsylvania, United States, is home to a plethora of hospitals, be it for the physically ill or, what was defined as the insane and feeble minded.

 In 1891, the St. Margaret Memorial Hospital opened under John H. Shoenberger.

 In 1908, the Pittsburgh Hospital was run under C. A. Fagan, and on June 1, 1889, the South Side Hospital opened under Anna M. Rindlaub.

 Pittsburgh, commonly known as the "Steel City," also built up a reputation for its abundance of hospitals that had opened up within a fairly close time period
of each other.

 During this time, many new mental hospitals were established due to overcrowding.

 In 1899, a building to house the mentally insane, Mayview State Hospital, was constructed; but it was not long before the complex had a reputation “as a place
of sorrow” due to the lack of proper health care patients received during their stay.
</item 10>
=== END ITEM id=971055e489d2d5f3 ===

=== ITEM id=f09bf00387e91336 ===
Gold sentences and facts:
Sentence 0:  The station is owned by Gray Television, as part of a duopoly with CBS affiliate KWCH-DT (channel 12); Gray also operates Univision affiliate KDCU-DT (channel 31) under a joint sales agreement with owner Entravision Communications Corporation.
  Fact 0.0: The station is owned by Gray Television.
  Fact 0.1: The station is part of a duopoly with CBS affiliate KWCH-DT (channel 12).
  Fact 0.2: Gray Television also operates Univision affiliate KDCU-DT (channel 31).
  Fact 0.3: Gray Television operates KDCU-DT under a joint sales agreement with its owner, Entravision Communications Corporation.
Sentence 1: KWCH-DT, virtual channel 12 (UHF digital channel 19), is a CBS-affiliated television station serving Wichita, Kansas, United States that is licensed to Hutchinson.
  Fact 1.0: KWCH-DT is virtual channel 12.
  Fact 1.1: KWCH-DT broadcasts on UHF digital channel 19.
  Fact 1.2: KWCH-DT is a CBS-affiliated television station.
  Fact 1.3: KWCH-DT serves Wichita, Kansas, United States.
  Fact 1.4: KWCH-DT is licensed to Hutchinson.

Retrieved items:
<item 1>
WFLD
WFLD, virtual channel 32 (UHF digital channel 31), is a Fox owned-and-operated television station licensed to Chicago, Illinois, United States.

 The station is owned by the Fox Television Stations division of 21st Century Fox, and operates as part of a television duopoly with Gary, Indiana-licensed
primary CW affiliate and secondary MyNetworkTV owned-and-operated station WPWR-TV (channel 50).

 The two stations share studio and office facilities located at Michigan Plaza on North Michigan Avenue in the Chicago Loop; WFLD's transmitter is based at the
Willis Tower on South Wacker Drive in the Loop business district.

 On cable, the station can be seen on Comcast Xfinity channel 12 in most parts of the Chicago area.
</item 1>

<item 2>
WJXX
WJXX, virtual channel 25 (VHF digital channel 10), is an ABC-affiliated television station serving Jacksonville, Florida, United States that is licensed to
Orange Park.

 The station is owned by the Tegna Media division of Tegna, Inc., as part of a duopoly with Jacksonville-licensed NBC affiliate WTLV (channel 12) (ironically a
former ABC affiliate itself from 1980 to 1988).

 The two stations share studio facilities on East Adams Street (near EverBank Field) in downtown Jacksonville; WJXX's transmitter is located on Eve Drive in the
city's Kilarney Shores section.
</item 2>

<item 3>
WDJT-TV
WDJT-TV, virtual channel 58 (UHF digital channel 46), is a CBS-affiliated television station licensed to Milwaukee, Wisconsin, United States.

 The station is owned by Chicago-based Weigel Broadcasting, as part of a duopoly with Racine-licensed independent station WMLW-TV (channel 49), and is also a
sister station to MeTV owned-and-operated station WBME-CD (channel 41, which is also carried via WDJT's DT2 subchannel) and Telemundo affiliate WYTU-LD (channel
63).

 All four stations share studio facilities located on South 60th Street in Milwaukee (near West Allis); WDJT's transmitter is located in Milwaukee's Lincoln
Park (next to the transmitter belonging to ABC affiliate WISN-TV, channel 12).
</item 3>

<item 4>
KXMB-TV
KXMB-TV, channel 12, is the CBS-affiliated television station for Bismarck, North Dakota.

 It broadcasts a high definition digital signal on VHF channel 12 (or virtual channel 12.1 via PSIP) from a transmitter near St. Anthony.

 The station can also be seen on Midcontinent cable channel 11 in Bismarck-Mandan, and cable channel 12 in most other areas.

 There is a high definition feed provided on Midcontinent digital channel 611.
</item 4>

<item 5>
KWCH-DT
KWCH-DT, virtual channel 12 (UHF digital channel 19), is a CBS-affiliated television station serving Wichita, Kansas, United States that is licensed to
Hutchinson.

 The station is owned by Gray Television, as part of a duopoly with CW affiliate KSCW-DT (channel 33); Gray also operates Univision affiliate KDCU-DT (channel
31) under a joint sales agreement with owner Entravision Communications Corporation.

 KWCH and KSCW share studio facilities and KDCU's master control operations are located on 37th Street in northeast Wichita; KWCH maintains transmitter
facilities located in rural northeastern Reno County (south-southeast of Buhler).
</item 5>

<item 6>
KMBC-TV
KMBC-TV, virtual channel 9 (UHF digital channel 29), is the ABC-affiliated television station licensed to Kansas City, Missouri, United States and also serving
Kansas City, Kansas.

 The station is owned by the Hearst Television subsidiary of the Hearst Corporation, as part of a duopoly with CW affiliate KCWE (channel 29).

 The two stations share studio facilities located at the Winchester Business Center on Winchester Avenue and East 63rd Street (near Swope Park, off of I-435) in
southeastern Kansas City, Missouri; KMBC-TV maintains transmitter facilities located near the Blue River in eastern Kansas City.

 On cable, KMBC-TV is available on Charter Spectrum, Comcast Xfinity and Consolidated Communications channel 12, and Google Fiber and AT&T U-verse channel 9.

 There is a high definition feed provided on Spectrum digital channel 1200, Xfinity channel 812, Consolidated channel 610 and U-verse channel 1009.
</item 6>

<item 7>
KDCU-DT
KDCU-DT, virtual and UHF digital channel 31, is a Univision-affiliated television station serving Wichita, Kansas, United States that is licensed to Derby.

 The station is owned by the Entravision Communications Corporation; Gray Television, which owns CBS affiliate KWCH-DT (channel 12) and CW affiliate KSCW-DT
(channel 33), operates KDCU under a joint sales agreement.

 KDCU's offices and master control operations are based at KWCH and KSCW's joint studio facilities on 37th Street in northeast Wichita; KDCU maintains
transmitter facilities located in rural northwestern Sedgwick County (northeast of Colwich).

 On cable, the station is available in standard definition on Cox Communications channel 15 and AT&T U-verse channel 31, and in high definition on Cox digital
channel 2015 and U-verse channel 1031.
</item 7>

<item 8>
KSCW-DT
KSCW-DT, virtual channel 33 (VHF digital channel 12), is a CW-affiliated television station licensed to Wichita, Kansas, United States.

 The station is owned by Gray Television, as part of a duopoly with CBS affiliate KWCH-DT (channel 12); Gray also operates Univision affiliate KDCU-DT (channel
31) under a joint sales agreement with owner Entravision Communications Corporation.

 KSCW and KWCH share studio facilities and KDCU's master control operations are located on East 37th Street North in northeastern Wichita; KSCW maintains
transmitter facilities located in rural northeastern Reno County (due south of Buhler).
</item 8>

<item 9>
KHSL-TV
KHSL-TV, virtual channel 12 (UHF digital channel 43) is a CBS-affiliated television station located in Chico, California, United States, that also serves
Redding.

 The station is owned by Heartland Media as part of a duopoly with NBC affiliate KNVN (channel 24), which is owned by Maxair Media but operated by Heartland
through a local marketing agreement.

 The two stations share studios at the corner of Eaton Road and Silverbell Road on the northwest side of Chico, while its transmitter is located along Cohasset
Road in rural Butte County northwest of Paradise.
</item 9>

<item 10>
WTLV
WTLV, virtual channel 12 (VHF digital channel 13), is an NBC-affiliated television station licensed to Jacksonville, Florida, United States.

 The station is owned by the Tegna Media division of Tegna, Inc., as part of a duopoly with Orange Park-licensed ABC affiliate WJXX (channel 25).

 The two stations share studio facilities on East Adams Street (near EverBank Field) in downtown Jacksonville; WTLV's transmitter is located on Eve Drive in the
city's Killarney Shores section.
</item 10>
=== END ITEM id=f09bf00387e91336 ===

=== ITEM id=65ffee201b33a4d7 ===
Gold sentences and facts:
Sentence 0: Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.
  Fact 0.0: Summer Magic is a 1963 Walt Disney Productions film.
  Fact 0.1: Summer Magic stars Hayley Mills.
  Fact 0.2: Summer Magic stars Burl Ives.
  Fact 0.3: Summer Magic stars Dorothy McGuire.
  Fact 0.4: Summer Magic is a story about a Boston widow and her children.
  Fact 0.5: In Summer Magic, the Boston widow and her children take up residence in a small town in Maine.
Sentence 1: Around the World in 80 Days is a 2004 American action-adventure comedy family film based on Jules Verne's novel of the same name.
  Fact 1.0: Around the World in 80 Days is a 2004 American film.
  Fact 1.1: Around the World in 80 Days is an action-adventure comedy family film.
  Fact 1.2: Around the World in 80 Days is based on Jules Verne's novel of the same name.

Retrieved items:
<item 1>
Summer Holiday (1963 film)
Summer Holiday is a British CinemaScope and Technicolor musical film featuring singer Cliff Richard.

 The film was directed by Peter Yates (his debut), produced by Kenneth Harper.

 The original screenplay was written by Peter Myers and Ronald Cass (who also wrote most of the song numbers and lyrics).

 The cast includes Lauri Peters, Melvyn Hayes, Teddy Green, Jeremy Bulloch, Una Stubbs, Pamela Hart, Jacqueline Daryl, Lionel Murton, Madge Ryan, David Kossoff,
Nicholas Phipps, Ron Moody and The Shadows.

 Herbert Ross choreographed the musical numbers.
</item 1>

<item 2>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 2>

<item 3>
Around the World (1956 song)
"Around the World" was the theme tune from the 1956 movie "Around the World in 80 Days" In the film, only an instrumental version of the song appeared, although
the vocal version has become by far the better known one.

 The song was written by Harold Adamson and Victor Young; Young died in 1956, several weeks after the film's release and he received the Academy Award for Best
Music, Scoring of a Dramatic or Comedy Picture posthumously.

 Young's orchestral version was a #13 hit on the Billboard charts in 1957.

 The recording by Bing Crosby was the B-side of the Victor Young version in 1957, on Festival SP45-1274 in Australia, and was a joint charting success.
</item 3>

<item 4>
Around the World in 80 Days (1956 film)
Around the World in 80 Days (sometimes spelled as Around the World in Eighty Days) is a 1956 American epic adventure-comedy film starring Cantinflas and David
Niven , produced by the Michael Todd Company and released by United Artists.
</item 4>

<item 5>
Michael Palin: Around the World in 80 Days
Around the World in 80 Days is a 7-part BBC television travel series first broadcast on BBC1 in 1989.

 It was presented by comedian and actor Michael Palin.

 The show was inspired by Jules Verne's classic novel "Around the World in Eighty Days", in which a character named Phileas Fogg accepts a wager to
circumnavigate the globe in eighty days or less.

 Palin was given the same deadline, and not allowed to use aircraft, which did not exist in Jules Verne's time and would make completing the journey far too
easy.

 He followed Phileas Fogg's route as closely as possible.

 Along the way he commented on the sights and cultures he encountered.

 Palin encountered several setbacks during his voyage, partly because he travelled with a five-person film crew, who are collectively named after Passepartout,
Phileas Fogg's manservant.
</item 5>

<item 6>
The Good Old Days (film)
The Good Old Days is a 1939 British comedy film directed by Roy William Neill.

 Written by Austin Melford and John Dighton based on a story by Ralph Smart, it stars Max Miller, Hal Walters and Kathleen Gibson.

 The film tells the story of group of entertainers struggling to obtain permission to perform at a tavern in 1840.
</item 6>

<item 7>
Around the World in 80 Days (1988 film)
Around the World in 80 Days is an Australian 48-minute direct-to-video animated film from Burbank Films Australia.

 It was originally released in 1988.

 The film is based on Jules Verne's classic French novel, "Around the World in 80 Days", first published in 1873, and was adapted by Leonard Lee.

 It was produced by Roz Phillips and featured original music by Simon Walker.

 The film imitated BRB Internacional's Spanish 1981 series, "La vuelta al mundo de Willy Fog", in its use of anthropomorphic animals in the human roles.

 The copyright in this film is now owned by Pulse Distribution and Entertainment and administered by digital rights management firm NuTech Digital.
</item 7>

<item 8>
Around the World in Eighty Days
Around the World in Eighty Days (French: "Le tour du monde en quatre-vingts jours" ) is a classic adventure novel by the French writer Jules Verne, published in
1873.

 In the story, Phileas Fogg of London and his newly employed French valet Passepartout attempt to circumnavigate the world in 80 days on a £20,000 wager (the
approximate equivalent of £2 million in 2016) set by his friends at the Reform Club.

 It is one of Verne's most acclaimed works .
</item 8>

<item 9>
Summer Magic
Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children
taking up residence in a small town in Maine.

 The film was based on the novel "Mother Carey's Chickens" by Kate Douglas Wiggin and was directed by James Neilson.

 This was the fourth of six film Mills did for Disney, and the young actress received a Golden Globe nomination for her work here.
</item 9>

<item 10>
Around the World in 80 Days (2004 film)
Around the World in 80 Days is a 2004 American action-adventure comedy family film based on Jules Verne's novel of the same name.

 It stars Jackie Chan, Steve Coogan and Cécile de France.

 The film is set in 19th-century Britain and centers on Phileas Fogg (Steve Coogan), here reimagined as an eccentric inventor, and his efforts to circumnavigate
the globe in 80 days.

 During the trip, he is accompanied by his Chinese valet, Passepartout (Jackie Chan).

 For comedic reasons, the film intentionally deviated wildly from the novel and included a number of anachronistic elements.

 With production costs of about $110 million and estimated marketing costs of $30 million, it earned $24 million at the U.S. box office and $72 million
worldwide, making it a box office flop.

 It was Arnold Schwarzenegger's last film before he took a hiatus from acting to become Governor of California until 2010's "The Expendables".
</item 10>
=== END ITEM id=65ffee201b33a4d7 ===

=== ITEM id=2c6b467fa4f3504a ===
Gold sentences and facts:
Sentence 0: Age of Chance were a British alternative rock-dance crossover band from Leeds, England active from 1985 to 1991.
  Fact 0.0: Age of Chance was a British alternative rock-dance crossover band.
  Fact 0.1: Age of Chance was from Leeds, England.
  Fact 0.2: Age of Chance was active from 1985 to 1991.
Sentence 1: Vector was a rock band formed in Sacramento, California in the early 1980s by Jimmy Abegg, Steve Griffith and Charlie Peacock.
  Fact 1.0: Vector was a rock band.
  Fact 1.1: Vector was formed in Sacramento, California.
  Fact 1.2: Vector was formed in the early 1980s.
  Fact 1.3: Vector was formed by Jimmy Abegg.
  Fact 1.4: Vector was formed by Steve Griffith.
  Fact 1.5: Vector was formed by Charlie Peacock.

Retrieved items:
<item 1>
New Valley LLC
New Valley LLC is an investment company owned by Vector Group.

 Western Union was renamed New Valley in the early 1990s as the corporate parent of all Western Union related businesses.

 In 1994, New Valley sold its Western Union businesses to First Financial Management (later acquired by First Data) for $1.2 Billion.

 After more than 100 years of being a publicly traded entity, New Valley became a wholly owned subsidiary of Vector Group on December 13, 2005 when Vector Group
acquired the remaining 42.3% of New Valley's common shares that it did not already own.

 New Valley presently has a 70% stake in Douglas Elliman Realty, the largest residential real estate brokerage in the New York metropolitan area.

 The company also has stakes in various hotels and golf courses in the United States.

 In conjunction with the spin-off of its Ladenburg Thalmann subsidiary New Valley retained a 7.6% stake in Ladenburg Thalmann Financial Services, an investment
bank and stock brokerage based in Miami.
</item 1>

<item 2>
Vector (band)
Vector was a rock band formed in Sacramento, California in the early 1980s by Jimmy Abegg, Steve Griffith and Charlie Peacock.

 The band had several drummers over the years, including Aaron Smith and Bruce Spencer, both of whom also played drums for The 77s.
</item 2>

<item 3>
Age of Chance
Age of Chance were a British alternative rock-dance crossover band from Leeds, England active from 1985 to 1991.

 They were perhaps most known for their mutant metallic cover of Prince's "Kiss" which topped the UK Indie Chart in 1986, and peaked at No. 50 in the UK Singles
Chart in January the following year.

 Despite signing for major label Virgin, and being favourites with the UK music press, they never enjoyed a major hit in the UK, although "Don't Get Mad… Get
Even" reached No. 8 on the US "Billboard" Hot Dance/Club Play chart.
</item 3>

<item 4>
Christopher Evans (author)
Christopher Evans (born 1951 in Tredegar, Wales) is a British science fiction writer and children's author.

 His novels include "Capella's Golden Eyes" (1980), "The Insider" (1981), "Mortal Remains" (1995) and "Ice Tower" (2000).

 He is the co-editor (with Robert Holdstock) of three original SF anthologies, "Other Edens" (1987), "Other Edens II" (1988) and "Other Edens III" (1989).
</item 4>

<item 5>
Transstadial transmission
Transstadial transmission occurs when a pathogen remains with the vector from one life stage ("stadium") to the next.

 For example, the bacteria "Borrelia burgdorferi", the causative agent for Lyme disease, infects the tick vector as a larva, and the infection is maintained
when it molts to a nymph and later develops as an adult.

 This type of transmission is seen in other parasites like viruses or "Rickettsia".

 In addition to ticks, mites are another common vector.

 Some sources consider transstadial transmission a type of horizontal transmission, whereas other sources consider it vertical or partial vertical transmission.
</item 5>

<item 6>
King Electric
King Electric is Chris Isepp and Peter Hartwig.

 They took their first musical attempts at the end of 1980s / beginning 1990s and first met each other at the upcoming and flourishing Viennese techno scene
back in '92.

 While Hartwig was touring with the well known noise-pop band the Basket Boys, Isepp made first attentions as Electro DJ & Producer.
</item 6>

<item 7>
A Wind in the Door
A Wind in the Door is a young adult science fantasy novel by Madeleine L'Engle.

 It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
</item 7>

<item 8>
Word2vec
Word2vec is a group of related models that are used to produce word embeddings.

 These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words.

 Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus
being assigned a corresponding vector in the space.

 Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the
space.
</item 8>

<item 9>
AES+F
AES+F is an artist collective of four Russian artists: Tatiana Arzamasova (1955), Lev Evzovich (1958), Evgeny Svyatsky (1957), and Vladimir Fridkes (1956).

 It was first formed as AES Group in 1987 by Arzamasova, Evzovich, and Svyatsky, becoming AES+F when Fridkes joined in 1995.

 The collective works in photography, video, installation, and animation, as well as more traditional media, such as painting, drawing, and sculpture.

 AES+F's early work included performance, installation, painting, and illustration.

 Well known for their monumental video-art installations that Gareth Harris describes as "monumental painting set in motion", AES+F create grand visual
narratives that explore contemporary global values, vices and conflicts.
</item 9>

<item 10>
Multivariate random variable
In probability, and statistics, a multivariate random variable or random vector is a list of mathematical variables each of whose value is unknown, either
because the value has not yet occurred or because there is imperfect knowledge of its value.

 The individual variables in a random vector are grouped together because they are all part of a single mathematical system — often they represent different
properties of an individual statistical unit.

 For example, while a given person has a specific age, height and weight, the representation of these features of "an unspecified person" from within a group
would be a random vector.

 Normally each element of a random vector is a real number.
</item 10>
=== END ITEM id=2c6b467fa4f3504a ===

=== ITEM id=a6ec12784f1ec4e7 ===
Gold sentences and facts:
Sentence 0: Awolnation is an American alternative rock band, formed and fronted by Aaron Bruno, formerly of Under the Influence of Giants, Home Town Hero, and Insurgence.
  Fact 0.0: Awolnation is an American alternative rock band.
  Fact 0.1: Awolnation was formed by Aaron Bruno.
  Fact 0.2: Awolnation is fronted by Aaron Bruno.
  Fact 0.3: Aaron Bruno was formerly of Under the Influence of Giants.
  Fact 0.4: Aaron Bruno was formerly of Home Town Hero.
  Fact 0.5: Aaron Bruno was formerly of Insurgence.
Sentence 1: This Mortal Coil was a music collective led by Ivo Watts-Russell, founder of the British record label 4AD.
  Fact 1.0: This Mortal Coil was a music collective.
  Fact 1.1: This Mortal Coil was led by Ivo Watts-Russell.
  Fact 1.2: Ivo Watts-Russell founded the British record label 4AD.

Retrieved items:
<item 1>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 1>

<item 2>
Kill Your Heroes
"Kill Your Heroes" is a song by American alternative rock band Awolnation.

 It is written by lead singer Aaron Bruno and musician Brian West for the band's debut studio album "Megalithic Symphony", where it appears as the eighth track.

 "Kill Your Heroes" was released as the third and final single from "Megalithic Symphony" and reached the top 20 of the United States "Billboard" Alternative
Songs and Rock Songs charts.
</item 2>

<item 3>
Awolnation discography
American alternative rock band Awolnation has released two studio albums, three extended plays, eight singles and ten music videos.

 The band is signed to Red Bull Records, and their first EP, "Back from Earth", was released on iTunes on May 18, 2010.

 They released their first studio album, "Megalithic Symphony", on March 15, 2011; it featured their most notable hit, "Sail", which peaked at #17 on the
"Billboard" Hot 100, #4 on the "Billboard" Rock Songs chart, and #5 on the "Billboard" Alternative Songs chart.

 The song has been certified 6× platinum by the RIAA and has sold 5,500,000 copies in the United States.

 As of February 29, 2016, the album has been certified platinum.

 The band's second studio album, "Run", was released on March 17, 2015.
</item 3>

<item 4>
Hollow Moon (Bad Wolf)
"Hollow Moon (Bad Wolf)" is a song by American alternative rock band Awolnation.

 It was released as the lead single from their second studio album "Run" (2015).

 The song became their first number-one hit on the Billboard Alternative Songs chart in early 2015.
</item 4>

<item 5>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 5>

<item 6>
Not Your Fault
"Not Your Fault" is a song by American alternative rock band Awolnation.

 It was originally written by frontman Aaron Bruno and recorded by the band for their debut studio album "Megalithic Symphony", where it appears as the twelfth
track.

 "Not Your Fault" was released to American modern rock radio on October 17, 2011, and was also released as a single in the United Kingdom and Bosnia on February
26, 2012.

 The single peaked at numbers three and eleven on the US "Billboard" Alternative Songs and Rock Songs charts, respectively.
</item 6>

<item 7>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 7>

<item 8>
This Mortal Coil
This Mortal Coil was a music collective led by Ivo Watts-Russell, founder of the British record label 4AD.

 Although Watts-Russell and John Fryer were technically the only two official members, the band's recorded output featured a large rotating cast of supporting
artists, many of whom were otherwise associated with 4AD, including members of Cocteau Twins, Colourbox, and Dead Can Dance.

 The project became known for its gothic, dream pop sound, and released three full albums beginning in 1984 with "It'll End in Tears".
</item 8>

<item 9>
Assembly of Dust
Assembly of Dust (also known as AOD) is an American rock band formed in 2002 by former Strangefolk frontman Reid Genauer.

 After the breakup of Strangefolk, Genauer decided to record a solo album, and he recruited some friends from Strangefolk's days on the road to help out.

 After titling the 2003 solo release “Assembly of Dust,” Genauer decided to use the name for his new group.

 AOD originally featured Genauer on lead vocals and guitar, Nate Wilson on the keyboard, Adam Terrell on lead guitar, John Leccese on bass, and Andy Herrick on
drums.

 In recent years the band has experienced some transformations, and the current status of the band now features Reid Genauer on lead vocals and guitar, Adam
Terrell on lead guitar, John Leccese on bass, Jason Crosby on the keys and violin, and Dave Diamond on the drums.

 Despite the changes in the band's lineup, AOD has managed to maintain the same high energy and tight musical finesse for which they are known.
</item 9>

<item 10>
Awolnation
Awolnation is an American alternative rock band, formed and fronted by Aaron Bruno, formerly of Under the Influence of Giants, Home Town Hero, and Insurgence.

 The band is signed to Red Bull Records, and their first EP, "Back from Earth", was released on iTunes on May 18, 2010.

 They released their first studio album, "Megalithic Symphony", on March 15, 2011; it featured their most notable hit, "Sail", which peaked at #17 on the
"Billboard" Hot 100, #4 on the "Billboard" Rock Songs chart, and #5 on the "Billboard" Alternative Songs chart.

 The song has been certified 6× platinum by the RIAA and has sold 5,500,000 copies in the United States.

 As of February 29, 2016, the album has been certified platinum.
</item 10>
=== END ITEM id=a6ec12784f1ec4e7 ===

=== ITEM id=5700ab2ed4731491 ===
Gold sentences and facts:
Sentence 0: Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.
  Fact 0.0: Wenling is a coastal county-level city.
  Fact 0.1: Wenling is in the municipal region of Taizhou.
  Fact 0.2: Taizhou is in southeastern Zhejiang province.
  Fact 0.3: Wenling is in China.
  Fact 0.4: In the Wenling dialect, Wenling is called Uen-lin Zy.
Sentence 1: Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.
  Fact 1.0: Ruili is a county-level city of Dehong Prefecture.
  Fact 1.1: Ruili is in the west of Yunnan province.
  Fact 1.2: Yunnan province is in the People's Republic of China.
  Fact 1.3: Ruili's Burmese name is ရွှေလီ.
  Fact 1.4: Ruili's Thai name is เมืองมาว.

Retrieved items:
<item 1>
Ruili Border Economic Cooperation Zone
Ruili Border Economic Cooperation Zone (RLBECZ) is a Chinese State Council-approved Industrial Park based in Ruili City, Dehong Prefecture, Yunnan, China,
founded in 1992 and was established to promote trade between China and Myanmar.

 The zone connects Ruili with Man-wing City.
</item 1>

<item 2>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 2>

<item 3>
Linzhou, Henan
Linzhou (), formerly Lin County (林县), is a county-level city in Anyang, Henan, China.

 Adjacent to Shanxi Province and Hebei Province, it is located in the north of Henan Province and at the eastern foot of the Taihang Mountains.

 It covers an area of 2046 square kilometers and has a population of about one million.
</item 3>

<item 4>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 4>

<item 5>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 5>

<item 6>
Wanding Border Economic Cooperation Zone
Wanding Border Economic Cooperation Zone (WTBECZ) is a Chinese State Council-approved Industrial Park based in Wanding Town, Ruili City, Dehong Prefecture,
Yunnan, China, founded in 1992 and was established to promote trade between China and Myanmar.

 The zone spans 6 km sq. and is focuses on developing trading, processing, agriculture resources and tourism.
</item 6>

<item 7>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 7>

<item 8>
Dali–Ruili Railway
The Dali–Ruili Railway or Darui Railway (), is a single-track electrified railroad under construction in Yunnan Province of Southwest China.

 The line is slated to run 336.39 km from Dali in central Yunnan to Ruili in southwestern Yunnan on the border with Myanmar.

 The line traverses rugged terrain, and bridges and tunnels will account for 75% of the total track length, including a 36-km tunnel through the Gaoligong
Mountains.

 Construction began in May 2011 and is scheduled to take six years.

 Cities and towns along route include Dali, Yangbi Yi Autonomous County, Yongping County, Baoshan, Mang City (also known as Mangshi or Dehong) and Ruili.
</item 8>

<item 9>
Ruili
Ruili (; Burmese: ရွှေလီ ; Thai: เมืองมาว ) is a county-level city of Dehong Prefecture, in the west of Yunnan province, People's Republic of China.

 It is a major border crossing between China and Myanmar, with the town of Muse located across the border.
</item 9>

<item 10>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 10>
=== END ITEM id=5700ab2ed4731491 ===
