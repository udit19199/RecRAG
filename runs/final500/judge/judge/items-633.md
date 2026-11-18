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

=== ITEM id=0c2d7c2b2f35c524 ===
Gold sentences and facts:
Sentence 0: Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.
  Fact 0.0: Charles Hastings Judd was a Colonel.
  Fact 0.1: Charles Hastings Judd was born on September 8, 1835.
  Fact 0.2: Charles Hastings Judd died on April 18, 1890.
  Fact 0.3: Charles Hastings Judd was an American businessman.
  Fact 0.4: Charles Hastings Judd was an American rancher.
  Fact 0.5: Charles Hastings Judd was an American courtier.
  Fact 0.6: Charles Hastings Judd was an American politician of the Kingdom of Hawaii.
  Fact 0.7: Charles Hastings Judd was a businessman, rancher, courtier and politician of the Kingdom of Hawaii.
Sentence 1:  He served as Chamberlain and Colonel of the military staff of King Kalākaua and traveled with the monarch on his 1881 world tour.
  Fact 1.0: Charles Hastings Judd served as Chamberlain to King Kalakaua.
  Fact 1.1: Charles Hastings Judd served as Colonel of the military staff of King Kalakaua.
  Fact 1.2: Charles Hastings Judd traveled with King Kalakaua on his 1881 world tour.
Sentence 2: Kalākaua (November 16, 1836 – January 20, 1891), born David Laʻ amea Kamananakapu Mahinulani Naloiaehuokalani Lumialani Kalākaua and sometimes called The Merrie Monarch, was the last king (and second to last monarch) of the Kingdom of Hawaiʻ i.
  Fact 2.0: Kalakaua was born on November 16, 1836.
  Fact 2.1: Kalakaua died on January 20, 1891.
  Fact 2.2: Kalakaua was born David La'amea Kamananakapu Mahinulani Naloiaehuokalani Lumialani Kalakaua.
  Fact 2.3: Kalakaua is sometimes called The Merrie Monarch.
  Fact 2.4: Kalakaua was the last king of the Kingdom of Hawaii.
  Fact 2.5: Kalakaua was the second-to-last monarch of the Kingdom of Hawaii.

Retrieved items:
<item 1>
HMS Queen Elizabeth March
"HMS Queen Elizabeth March", composed by WO2 bandmaster John Morrish RM, was the winning composition for the 2012 Royal Marines Band Service March Competition,
sponsored by the Aircraft Carrier Alliance.

 The march was written for the first of the new generation, of Royal Navy Queen Elizabeth-class aircraft carriers.
</item 1>

<item 2>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 2>

<item 3>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
</item 3>

<item 4>
Corps of Military Staff Clerks
The Corps of Military Staff Clerks was an administrative corps of the Canadian Army.

 The Headquarters Corps of Military Staff Clerks was established in 1905.
</item 4>

<item 5>
Algernon Willis
Admiral of the Fleet Sir Algernon Usborne Willis (17 May 1889 – 12 April 1976) was a Royal Navy officer.

 He served in the First World War and saw action at the Battle of Jutland in May 1916.

 He also served in the Second World War as Commander-in-Chief, South Atlantic in which capacity he led actions against German and Japanese raiding ships.

 He continued his war service as Flag Officer commanding 3rd Battle Squadron and Second in command of the Eastern Fleet and then as Flag Officer commanding
Force H, the force which covered North African Operations, the Allied invasion of Sicily in July 1943 and then the Allied invasion of Italy in September 1943.

 He spent the final years of the war as Commander-in-Chief, Levant, in which capacity he conducted naval operations in support of the Dodecanese Campaign, and
then as Second Sea Lord, in which capacity he arranged the manpower for the campaign in the Pacific Ocean against the Imperial Japanese Navy.

 After the war he served as Commander-in-Chief, Mediterranean Fleet, in which role he was faced with unrest in Mandatory Palestine, before he became
Commander-in-Chief, Portsmouth.
</item 5>

<item 6>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 6>

<item 7>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 7>

<item 8>
Arthur Tedder, 1st Baron Tedder
Marshal of the Royal Air Force Arthur William Tedder, 1st Baron Tedder, {'1': ", '2': ", '3': ", '4': "} (11 July 1890 – 3 June 1967) was a senior Royal Air
Force commander.

 He was a pilot and squadron commander in the Royal Flying Corps in the First World War and he went on to serve as a senior officer in the Royal Air Force
during the inter-war years when he served in Turkey, Great Britain and the Far East.

 During the Second World War, as Air Officer Commanding RAF Middle East Command, Tedder directed air operations in the Mediterranean and North Africa, including
the evacuation of Crete and "Operation Crusader" in North Africa.

 His bombing tactics became known as the "Tedder Carpet".

 Later in the war Tedder took command of Mediterranean Air Command and in that role was closely involved in the planning of the Allied invasion of Sicily and
then the Allied invasion of Italy.

 When Operation Overlord—the invasion of France—came to be planned, Tedder was appointed Deputy Supreme Commander at Supreme Headquarters Allied Expeditionary
Force under General Eisenhower.
</item 8>

<item 9>
Military reference covering organizations, ranks and honors: Canada’s Staff Clerks Corps dates to 1905; the Ottoman college in Istanbul trained staff officers
in a two-year program. Military staffs manage administration, operations, logistics and information. Germany’s merit-based General Staff was formalized in 1814;
Vietnam’s was founded in 1945. Canada’s 2004 General Campaign Star recognizes campaign service. Ranks discussed include corporal, North Korean chŏnsa, Polish
podpor
</item 9>

<item 10>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 10>
=== END ITEM id=0c2d7c2b2f35c524 ===

=== ITEM id=cae0f94cddc3c7b1 ===
Gold sentences and facts:
Sentence 0:  Upon Victoria's death, the British throne passed to her eldest son Edward VII, a member of the House of Saxe-Coburg and Gotha through his father.
  Fact 0.0: Victoria died.
  Fact 0.1: The British throne passed to Victoria's eldest son Edward VII upon her death.
  Fact 0.2: Edward VII was a member of the House of Saxe-Coburg and Gotha through his father.
Sentence 1: The House of Saxe-Coburg and Gotha ( ; German: "") is a German dynasty that ruled the duchy of Saxe-Coburg and Gotha, which was one of the Ernestine duchies.
  Fact 1.0: The House of Saxe-Coburg and Gotha is a German dynasty.
  Fact 1.1: The House of Saxe-Coburg and Gotha ruled the duchy of Saxe-Coburg and Gotha.
  Fact 1.2: The duchy of Saxe-Coburg and Gotha was one of the Ernestine duchies.

Retrieved items:
<item 1>
Line of succession to the former Romanian throne
The succession order to the throne of the Romanian monarchy, abolished since 1947, was regulated by the monarchical constitution of 1923 and the 1884 Law of the
Romanian Royal House Rules enacted pursuant to the 1866 Constitution of Romania which had confirmed the enthronement of Prince Karl (Carol) of
Hohenzollern-Sigmaringen.

 The defunct 1923 Constitution stipulated Salic law, according to which the throne was hereditary in King Carol I's legitimate descent and, if his male issue
failed, in the descent of his brothers of the Sigmaringen princely branch of the House of Hohenzollern, according to agnatic primogeniture and to the perpetual
exclusion of females and their descendants.

 The last monarch to reign in Romania was King Michael I, who was born in 1921, abdicated his throne on 30 December 1947 under coercion, and went into exile in
Switzerland.
</item 1>

<item 2>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 2>

<item 3>
Canada under British rule
Although confederation in 1867 led to an enlarged Dominion with increased autonomy over domestic affairs, Canada still remained a colony within the British
Empire and was thus subordinate to the British Parliament until the enactment of the Statute of Westminster in 1931.

 This statute recognized Canada as an independent peer coequal with the United Kingdom, and thus provided the Parliament of Canada with legislative sovereignty
over all federal matters except the power to change the constitutional laws of Canada which remained under the purview of the Parliament of the United Kingdom.

 Canada's final vestige of legal dependence on the United Kingdom was terminated in 1982 with the enactment of the Canada Act, subsequently providing Canada
with full legal sovereignty completely independent of the United Kingdom.
</item 3>

<item 4>
House of Saxe-Coburg and Gotha
The House of Saxe-Coburg and Gotha ( ; German: "") is a German dynasty that ruled the duchy of Saxe-Coburg and Gotha, which was one of the Ernestine duchies.

 It is a cadet branch of the Saxon House of Wettin.
</item 4>

<item 5>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 5>

<item 6>
House of Oldenburg
The House of Oldenburg is a European dynasty of North German origin.

 It is one of Europe's most influential royal houses, with branches that rule or have ruled in Denmark, Iceland, Greece, Norway, Russia, Sweden, Schleswig,
Holstein, and Oldenburg.

 The current Queen of Denmark and King of Norway, the former King of Greece, the consorts of Greece and the United Kingdom, as well as the first twelve names in
the line of succession to the British throne, are all patrilineal members of the Glücksburg branch of this house.
</item 6>

<item 7>
Principality of Calenberg
The Principality of Calenberg was a dynastic division of the Welf duchy of Brunswick-Lüneburg established in 1432.

 Calenberg was ruled by the House of Hanover from 1635 onwards; the princes received the ninth electoral dignity of the Holy Roman Empire in 1692.

 Their territory became the nucleus of the Electorate of Hanover, ruled in personal union with the Kingdom of Great Britain from 1714 onwards.

 The principality received its name from Calenberg Castle, a residence of the Brunswick dukes.
</item 7>

<item 8>
Succession to the British throne
Succession to the British throne is determined by descent, gender (for people born before October 2011), legitimacy, and religion.

 Under common law, the Crown is inherited by a sovereign's children or by a childless sovereign's nearest collateral line.

 The Bill of Rights 1689 and the Act of Settlement 1701, restrict succession to the throne to the legitimate Protestant descendants of Sophia of Hanover that
are in "communion with the Church of England".

 Spouses of Roman Catholics were disqualified from 1689 until the law was amended in 2015.

 Protestant descendants of those excluded for being Roman Catholics are eligible.
</item 8>

<item 9>
House of Hanover
The House of Hanover (or the Hanoverians ; German: "Haus Hannover" ) is a German royal dynasty that ruled the Electorate and then the Kingdom of Hanover, and
that also provided monarchs of Great Britain and Ireland from 1714 and ruled the United Kingdom until the death of Queen Victoria in 1901.

 Upon Victoria's death, the British throne passed to her eldest son Edward VII, a member of the House of Saxe-Coburg and Gotha through his father.
</item 9>

<item 10>
Duke of Bedford
He regained the earldom in 1485 when his nephew Henry VII came to the throne and was elevated to the dukedom the same year.

 He had no legitimate children and the titles became extinct on his death in 1495.
</item 10>
=== END ITEM id=cae0f94cddc3c7b1 ===

=== ITEM id=168c5dccac2cce79 ===
Gold sentences and facts:
Sentence 0:  is a board game created by Glenn Drover and published by Eagle Games in 2003.
  Fact 0.0: The board game in this sentence was created by Glenn Drover.
  Fact 0.1: The board game in this sentence was published by Eagle Games in 2003.
Sentence 1: Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.
  Fact 1.0: Can't Stop is a board game designed by Sid Sackson.
  Fact 1.1: Can't Stop was originally published by Parker Brothers in 1980.
  Fact 1.2: Can't Stop was long out of print in the United States.

Retrieved items:
<item 1>
TriBond
TriBond is a board game that has sold over 3 million copies in 14 countries since its release in 1990.

 It requires players to determine a common bond between three subjects.

 It follows in the tradition of "Trivial Pursuit", "Outburst" and other adult boardgames that require a wide range of knowledge but "TriBond" requires some
problem solving ability as well.
</item 1>

<item 2>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 2>

<item 3>
Francis Tresham (game designer)
Francis Tresham is a United Kingdom-based board game designer who has been producing board games since the early 1970s.

 Tresham founded and ran games company Hartland Trefoil (founded 1971), a company well known for its "Civilization" board game, until its sale to MicroProse in
1997.

 His "1829" game was the first of the "18xx" board game series and some of his board games have inspired Sid Meier computer games such as "Railroad Tycoon".
</item 3>

<item 4>
Plaid Hat Games
Plaid Hat Games is a United States-based board game studio.

 Plaid Hat Games was founded in 2009.

 Board game designer Colby Dauch formed a board game publishing company in order to release the companies first game, Summoner Wars.
</item 4>

<item 5>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 5>

<item 6>
SpongeBob SquarePants: Battle for Bikini Bottom
SpongeBob SquarePants: Battle for Bikini Bottom is a 2003 video game based on the animated series of the same name, developed by Heavy Iron Studios, AWE Games,
and Vicarious Visions, and published by THQ.

 The game was released for the PlayStation 2, Game Boy Advance (GBA), Xbox, and Nintendo GameCube consoles as well as a port for Microsoft Windows.

 The Microsoft Windows version of the game is a point-and-click adventure game, whereas the 6th generation platforms received a 3D platformer.

 The Game Boy Advance version received a 2D platformer.

 All versions of the game feature an original storyline, in which the player attempts to defend Bikini Bottom from an invasion of robots created by Plankton
with a machine called the Duplicatotron 3000, playing as SpongeBob in all versions, as well as Patrick and Sandy in the console versions.

 The game was released on October 31, 2003, in North America and in Europe on November 28, 2003.
</item 6>

<item 7>
Uncle Wiggily (board game)
Uncle Wiggily Game is a track board game based on a character in a series of children's books by American writer Howard Roger Garis.

 The game is of the "racing" variety in the style of the European "Goose Game".

 Players advance along the track from Uncle Wiggily's Bungalow to Dr. Possum's House.

 There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.

 The game was first published by Milton Bradley in 1916 and has seen several editions with minor modifications over the years.

 "Uncle Wiggily" remains one of the first and favorite games of childhood, and, with "Candy Land", is considered a classic juvenile American board game.
</item 7>

<item 8>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 8>

<item 9>
Castle Ravenloft Board Game
Castle Ravenloft Board Game is a 2010 board game published by Wizards of the Coast.

 It was the first game released in the Dungeons & Dragons Adventure System board game series.
</item 9>

<item 10>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 10>
=== END ITEM id=168c5dccac2cce79 ===

=== ITEM id=679fcbc9ee8ac909 ===
Gold sentences and facts:
Sentence 0: The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.
  Fact 0.0: The Bears and I is a 1974 American drama film.
  Fact 0.1: The Bears and I was directed by Bernard McEveety.
  Fact 0.2: The Bears and I was written by John Whedon.
Sentence 1:  The film was released on July 31, 1974, by Buena Vista Distribution.
  Fact 1.0: The Bears and I was released on July 31, 1974.
  Fact 1.1: The Bears and I was released by Buena Vista Distribution.
Sentence 2: Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as co-director.
  Fact 2.0: Oceans (French: Océans) is a 2009 French nature documentary film.
  Fact 2.1: Oceans was directed by Jacques Perrin.
  Fact 2.2: Oceans was produced by Jacques Perrin.
  Fact 2.3: Oceans was co-written by Jacques Perrin.
  Fact 2.4: Oceans was narrated by Jacques Perrin.
  Fact 2.5: Jacques Cluzaud was co-director of Oceans.

Retrieved items:
<item 1>
The Best of Creedence Clearwater Revival
The Best of Creedence Clearwater Revival is a compilation album by American rock band Creedence Clearwater Revival, released in 1977.

 The album features all the tracks of Chronicle which was released a year earlier in 1976, as well as "Good Golly Miss Molly", "Born on the Bayou", "Cotton
Fields", "Hello Mary Lou", "The Midnight Special", "Walk on the Water", and "Night Time Is the Right Time" which would appear on the 1986 release "Chronicle,
Vol.

 2", and "Bootleg" which does not appear on either volume.
</item 1>

<item 2>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 2>

<item 3>
Foxes (soundtrack)
Foxes is the soundtrack to the 1980 film of the same name, starring Jodie Foster, Scott Baio, Sally Kellerman, Randy Quaid as well as The Runaways' lead singer
Cherie Currie.

 The double-album was released on the disco label Casablanca Records.
</item 3>

<item 4>
Buena Vista (brand)
Buena Vista is a brand name which was historically often used for divisions and subsidiaries of The Walt Disney Company, whose primary studios, the Walt Disney
Studios, are located on Buena Vista Street in Burbank, California.

 (The studio lot is also home to the Company's corporate headquarters, the Team Disney Burbank building.)

 The logos for the various Buena Vista brands featured the "Buena Vista" wordmark superimposed over the Disney Castle image to signal the affiliation between
Buena Vista and Disney.

 The name literally means "good view" in Spanish.
</item 4>

<item 5>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 5>

<item 6>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 6>

<item 7>
Oceans (film)
Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as
co-director.

 The film explores the marine species of Earth's five oceans and reflects on the negative aspects of human activity on the environment, with Perrin (Pierce
Brosnan in English) providing narration.
</item 7>

<item 8>
Return to the Blue Lagoon
Return to the Blue Lagoon is a 1991 American romance and adventure film starring Milla Jovovich and Brian Krause, produced and directed by William A. Graham.

 The film is a sequel to "The Blue Lagoon" (1980).

 The screenplay by Leslie Stevens was based on the novel "The Garden of God" by Henry De Vere Stacpoole.

 The original music score was composed by Basil Poledouris.

 The film's closing theme song "A World of Our Own" is performed by Surface featuring Bernard Jackson.

 The music was written by Barry Mann, and the lyrics were written by Cynthia Weil.
</item 8>

<item 9>
I Against I
I Against I is the third studio album by the American hardcore punk band Bad Brains.

 It was released in November 1986 through SST Records with the catalog number SST 065.

 The best-selling album in the band's catalog, "I Against I" is a critically acclaimed album that mixes American hardcore punk with funk, soul, reggae and heavy
metal.

 It is also included in the book "1001 Albums You Must Hear Before You Die".

 The album featured an MTV video for the title track directed by Paul Rachman who later went on to direct the documentary feature film "American Hardcore".
</item 9>

<item 10>
Entre el Cielo y el Infierno
Entre el Cielo y el Infierno (Between Heaven and Hell) is the fifth studio album by Argentine heavy metal band Rata Blanca.

 The album was recorded at Estudios La Nave between June and July ,1994 and was released in September, 1994 by BMG.

 Is the first and unique album featuring vocalist Mario Ian and also is the first album featuring Javier Retamozo on keyboards.
</item 10>
=== END ITEM id=679fcbc9ee8ac909 ===

=== ITEM id=a14cce1930352c34 ===
Gold sentences and facts:
Sentence 0:  She is a member of the "Game of Thrones" stunt teams that won two Screen Actors Guild Awards, and has done stunt performances for films including "Rogue One".
  Fact 0.0: The person is a member of the Game of Thrones stunt teams.
  Fact 0.1: The Game of Thrones stunt teams won two Screen Actors Guild Awards.
  Fact 0.2: The person has done stunt performances for films including Rogue One.
Sentence 1:  "Rogue One" follows a group of rebels on a mission to steal the plans for the Death Star, the Galactic Empire's superweapon.
  Fact 1.0: Rogue One follows a group of rebels.
  Fact 1.1: The rebels are on a mission to steal the plans for the Death Star.
  Fact 1.2: The Death Star is the Galactic Empire's superweapon.

Retrieved items:
<item 1>
Rachelle Beinart
Rachelle Amy Beinart (born 29 September 1984) is a British actress and stunt performer.

 She is a member of the "Game of Thrones" stunt teams that won two Screen Actors Guild Awards, and has done stunt performances for films including "Rogue One".

 She has also acted in preschool children's shows as a suit actor, including the reboot of "Teletubbies".
</item 1>

<item 2>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 2>

<item 3>
Rebecca Forstadt
Rebecca Lynn Forstadt (born December 16, 1953), also known as Reba West, is an American voice actress, best known for playing young female roles in various
animated series.

 After studying theater at Orange Coast College, in Costa Mesa, California, Forstadt began her acting career by working at Knott's Berry Farm's Bird Cage
Theater, performing melodramas, often as the damsel in distress character.

 Later, she went to Hollywood where she worked as a wardrobe mistress on such television shows as "The White Shadow" and "Hill Street Blues", as well as for the
film "S.O.B.".

 She also spent several years doing live theater in the Los Angeles area.

 Most notably, she won some recognition for her portrayal of the character Josette in the world premiere of Eugène Ionesco's "Tales for People Under 3 Years of
Age" at the Stages Theatre Center in 1982.

 She starred in several low-budget movies such as "Mugsy's Girls", with Ruth Gordon and Laura Branigan, and "Round Numbers" with Kate Mulgrew, Samantha Eggar,
and Shani Wallis.
</item 3>

<item 4>
Contraband (2012 film)
Contraband is a 2012 American crime thriller film directed by Baltasar Kormákur, starring Mark Wahlberg, Kate Beckinsale, Ben Foster, Caleb Landry Jones,
Giovanni Ribisi, Lukas Haas, Diego Luna and J. K. Simmons.

 The film is a remake of the 2008 Icelandic film "Reykjavík-Rotterdam" which Baltasar Kormákur starred in.

 It was released on January 13, 2012 in the United States by Universal Pictures.
</item 4>

<item 5>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 5>

<item 6>
Rogue One
Rogue One: A Star Wars Story, or simply Rogue One, is a 2016 American epic space opera film directed by Gareth Edwards.

 The screenplay by Chris Weitz and Tony Gilroy is from a story by John Knoll and Gary Whitta.

 It was produced by Lucasfilm and distributed by Walt Disney Studios Motion Pictures.

 It is the first installment of the "Star Wars" "Anthology" series, set immediately before the events of the original "Star Wars" film.

 The cast includes Felicity Jones, Diego Luna, Riz Ahmed, Ben Mendelsohn, Donnie Yen, Mads Mikkelsen, Alan Tudyk, Jiang Wen and Forest Whitaker.

 "Rogue One" follows a group of rebels on a mission to steal the plans for the Death Star, the Galactic Empire's superweapon.
</item 6>

<item 7>
Rachelle Henry
Rachelle Henry (born December 16, 2000) is an American teen actress and filmmaker.

 She played the role of Sandy Hobbs in the TLC (TV network) Series "Escaping the Prophet"and Lissa Golaski in "Depth", the film prequel to the Soma (video game)
by Frictional Games.

 She is also known for directing and producing short films containing messages of social influence and coming of age themes including "MISSING" and "Almost
Boyfriends".
</item 7>

<item 8>
The Runaways (film)
The Runaways is a 2010 American drama film about the 1970s rock band of the same name written and directed by Floria Sigismondi.

 It is based on the book "Neon Angel: A Memoir of a Runaway" by the band's original lead vocalist Cherie Currie.

 The film stars Dakota Fanning as Currie, Kristen Stewart as rhythm guitarist and vocalist Joan Jett, and Michael Shannon as record producer Kim Fowley.

 "The Runaways" depicts the formation of the band in 1975 and focuses on the relationship between Currie and Jett until Currie's departure from the band.

 The film grossed about $4.7 million worldwide and received generally favorable reviews from critics.
</item 8>

<item 9>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 9>

<item 10>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 10>
=== END ITEM id=a14cce1930352c34 ===

=== ITEM id=290bfaec0280c200 ===
Gold sentences and facts:
Sentence 0:  Obertraun is a popular holiday destination offering activities such as skiing, snowboarding in the winter and mountain biking, swimming and kayaking in the summer.
  Fact 0.0: Obertraun is a popular holiday destination.
  Fact 0.1: Obertraun offers skiing and snowboarding in the winter.
  Fact 0.2: Obertraun offers mountain biking in the summer.
  Fact 0.3: Obertraun offers swimming and kayaking in the summer.
Sentence 1:  The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the precious salt mines of the Habsburg Monarchy.
  Fact 1.0: The name 'Salzkammergut' literally means 'Estate of the Salt Chamber'.
  Fact 1.1: The name 'Salzkammergut' derives from the Imperial Salt Chamber.
  Fact 1.2: The Imperial Salt Chamber was the authority charged with running the precious salt mines of the Habsburg Monarchy.

Retrieved items:
<item 1>
Hundsheimer Berg
Hundsheimer Berg is a hill located in the Hundsheimer Berge hill range in Lower Austria, Austria close to the border with Slovakia.

 Its peak is 481 m above sea level which makes it the highest hill in the Hundsheimer Berge.

 The north slopes of the Hundsheimer Berg are covered with forests while the southern slopes have a steppe vegetation.

 Since 1965, a nature reserve with the same name, "Hundsheimer Berg", is located on these southern slopes.
</item 1>

<item 2>
Monochrom’s fictional Soviet Unterzögersdorf, inspired by creator Johannes Grenzfurthner’s family stories, imagines the USSR’s last enclave in Austria. Liberal
arts were skills for civic participation. The Eastern Bloc comprised Soviet-aligned socialist states; the Warsaw Pact was its military alliance. After WWII,
about 31 million people, including Germans, were displaced across Central and Eastern Europe. The Curzon Line informed Poland’s eastern-border debate; Kresy,
incorporated into Soviet republics
</item 2>

<item 3>
Obertraun
Obertraun, Upper Austria is a village in the Salzkammergut, a region in Austria.

 It is located near the Hallstätter See (Hallstatt Lake) and Hoher Dachstein.

 Obertraun is a popular holiday destination offering activities such as skiing, snowboarding in the winter and mountain biking, swimming and kayaking in the
summer.
</item 3>

<item 4>
Upper Saxon German
Upper Saxon (German: "Obersächsisch" ) is an East Central German dialect spoken in much of the modern German State of Saxony and in the adjacent parts of
Saxony-Anhalt and Thuringia.

 Though colloquially called "Saxon" ("Sächsisch" ), it is not to be confused with the Low Saxon dialect group in Northern Germany.

 Upper Saxon is closely linked to the Thuringian dialect spoken in the adjacent areas to the west.
</item 4>

<item 5>
Appenzell Wars (1401–29): farmers and St. Gallen craftspeople challenged Habsburg and abbey rule. Appenzell Alps span three Swiss cantons. Landes de Gascogne
protects French pine forests, wetlands, and coast. Hungary’s 14-km² Lake Fehér, a protected national-park salt lake, shelters 280 bird species and migrants.
Upper Saxon and South Bergish are distinct German dialect groups. Austria’s UNESCO-listed
</item 5>

<item 6>
Salzkammergut
The Salzkammergut is a resort area located in Austria.

 It stretches from the City of Salzburg eastwards along the Austrian Alpine Foreland and the Northern Limestone Alps to the peaks of the Dachstein Mountains,
spanning the federal states of Upper Austria, Salzburg, and Styria.

 The main river of the region is the Traun, a right tributary of the Danube.

 The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the
precious salt mines of the Habsburg Monarchy.

 It is a UNESCO World Heritage Site.
</item 6>

<item 7>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
</item 7>

<item 8>
List of Austrian mountain climbers
This is a list of Austrian mountaineers.

 On 5 of the 14 Eight-thousanders Austrians have made the first ascent, more than any other nation can claim.

 Out of the total of 9 Austrian mountaineers who made first ascents of Eight-thousanders, 3 have been members of the Edelweiss Club Salzburg, an association of
mountaineers founded in Salzburg in 1881
</item 8>

<item 9>
South Bergish
South Bergish (German: "Südbergische Dialekte ") or Upper Bergish (German: "Oberbergische Dialekte ") is a group of German dialects of the Bergisches Land
Region East of the Rhine and approximately south of the Wupper and north of the Sieg.

 These dialects are part of the Ripuarian group and thus are also called East Ripuarian.

 Ripuarian dialects are also spoken west of the Rhine up to the German border, and in some small areas next to the respective borders in Belgium and in the
Netherlands.

 Ripuarian Bergish dialects belong to the Middle German group, and thus are varieties of High German, where they belong to the northmost ones.

 In the North, they border to the East Bergish and the West Bergish aka Low Bergish language groups, which are part of the Low Franconian group like Dutch.
</item 9>

<item 10>
Appenzell Alps
The Appenzell Alps (German: "Appenzeller Alpen" ) are a mountain range in Switzerland on the northern edge of the Alps.

 They extend into the cantons of Appenzell Ausserrhoden, Appenzell Innerrhoden and St. Gallen and are bordered by the Glarus Alps to the west and the Rätikon to
the south-east.
</item 10>
=== END ITEM id=290bfaec0280c200 ===

=== ITEM id=01b8a7bbf6e38a6f ===
Gold sentences and facts:
Sentence 0: Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
  Fact 0.0: Leycesteria is a genus of flowering plants.
  Fact 0.1: Leycesteria is in the honeysuckle family Caprifoliaceae.
  Fact 0.2: Leycesteria is native to temperate Asia in the Himalaya.
  Fact 0.3: Leycesteria is native to southwestern China.
Sentence 1: Anigozanthos is a small genus of Australian plants in the Bloodwort family Haemodoraceae.
  Fact 1.0: Anigozanthos is a small genus of Australian plants.
  Fact 1.1: Anigozanthos is in the Bloodwort family Haemodoraceae.

Retrieved items:
<item 1>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 1>

<item 2>
Xanthosoma
Xanthosoma is a genus of flowering plants in the arum family, Araceae.

 The genus is native to tropical America but widely cultivated and naturalized in other tropical regions.

 Several are grown for their starchy corms, an important food staple of tropical regions, known variously as "malanga", "otoy", "otoe", cocoyam (or new
cocoyam), "tannia", "tannier", "yautía", "macabo", "ocumo", "macal", "taioba", "dasheen", "quequisque", "ʻ ape" and (in Papua New Guinea) as Singapore taro
("taro kongkong").

 Many other species (including especially "X. roseum") are used as ornamental plants, and in popular horticultural literature are known as ‘ape or elephant ear
(from the purported resemblance of the leaf to an elephant's ear), although the latter name is sometimes also applied to members with similar appearance and
uses in the closely related genera "Caladium", "Colocasia" (i.e., taro), and "Alocasia".
</item 2>

<item 3>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 3>

<item 4>
Anigozanthos flavidus
Anigozanthos flavidus is a species of plant found in Southwest Australia.

 It is member of the Haemodoraceae family.

 It is commonly known as the tall, yellow, or evergreen, kangaroo paw.

 The specific epithet, "flavidus", refers to the yellow flowers of this plant.
</item 4>

<item 5>
Anigozanthos humilis
Anigozanthos humilis is a species of "Anigozanthos" in the family Haemodoraceae, known as common cat's paw or catspaw.

 This flowering perennial plant is endemic to Southwest Australia and widespread in its open forests.
</item 5>

<item 6>
Anigozanthos manglesii
Anigozanthos manglesii, commonly known as the red-and-green kangaroo paw, Mangles kangaroo paw, Kurulbrang(Noongar) is a plant species endemic to Western
Australia, and the floral emblem of that state.
</item 6>

<item 7>
Anigozanthos preissii
Anigozanthos preissii, the Albany cat's paw, is a herbaceous plant species in the family Haemodoraceae, endemic to Western Australia.
</item 7>

<item 8>
Mulga Island
Mulga Island is a small island 3 nmi off the coast and 5 nmi northeast of Kirkby Head, Enderby Land in Antarctica.

 Plotted from air photos taken from ANARE (Australian National Antarctic Research Expeditions) aircraft in 1956.

 Mulga is the vernacular name for species of Acacia found in semi-desert areas of Australia.
</item 8>

<item 9>
Leycesteria formosa
Leycesteria formosa (Himalayan honeysuckle, flowering nutmeg, Himalaya nutmeg or pheasant berry) is a deciduous shrub in the family Caprifoliaceae, native to
the Himalaya and southwestern China.

 It is considered a noxious invasive species in Australia, New Zealand, the neighboring islands of Micronesia, and some other places.
</item 9>

<item 10>
Anigozanthos
Anigozanthos is a small genus of Australian plants in the Bloodwort family Haemodoraceae.

 The 11 species and several subspecies are commonly known as kangaroo paw and catspaw depending on the shape of their flowers.

 A further species, previously identified as "Anigozanthos fuliginosus" and commonly known as the black kangaroo paw, has been transferred to its own monotypic
genus and is now known as "Macropidia fuliginosa".
</item 10>
=== END ITEM id=01b8a7bbf6e38a6f ===

=== ITEM id=3bbf4ca044f77159 ===
Gold sentences and facts:
Sentence 0: Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
  Fact 0.0: Grantley Jonathan "Grant" Nicholas was born on 12 November 1967.
  Fact 0.1: Grant Nicholas is a Welsh musician.
  Fact 0.2: Grant Nicholas is best known as the lead singer of the rock band Feeder.
  Fact 0.3: Grant Nicholas is best known as the lead guitarist of the rock band Feeder.
Sentence 1: Danny Shirley (born August 12, 1956) is an American country music singer.
  Fact 1.0: Danny Shirley was born on August 12, 1956.
  Fact 1.1: Danny Shirley is an American country music singer.

Retrieved items:
<item 1>
Nat and Alex Wolff
Nat and Alex Wolff are an American pop music duo from New York, New York, consisting of brothers Nat and Alex Wolff.

 The siblings are known for their work on the Nickelodeon television series "The Naked Brothers Band", which was created and produced by their mother, actress
Polly Draper.

 It was adapted from the mockumentary film that Draper wrote and directed.

 The duo's initial teen pop boy band called "The Naked Brothers Band" was depicted as part of its participation in their Nickelodeon series that aired from 2007
to 2009.

 They have released four full-length albums and have been nominated for several awards.
</item 1>

<item 2>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 2>

<item 3>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 3>

<item 4>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 4>

<item 5>
Fayard Nicholas
Fayard Antonio Nicholas (October 20, 1914 – January 24, 2006) was an American choreographer, dancer and actor.

 He and his younger brother Harold Nicholas made up the Nicholas Brothers tap-dance duo, who starred in the MGM musicals "An All-Colored Vaudeville Show"
(1935), "Stormy Weather" (1943), "The Pirate" (1948), and "Hard Four" (2007).

 The Nicholas brothers also starred in the 20th Century-Fox musicals "Down Argentine Way" (1940), "Sun Valley Serenade" (1941), and "Orchestra Wives" (1942).
</item 5>

<item 6>
Don't Stop Believing (TV series)
Don't Stop Believing is a British television talent show that aired on Channel 5 in summer 2010.

 It was inspired by the musical comedy-drama "Glee", which airs in the United States on the Fox network.

 The series featured live shows in which musical performance groups competed against each other, with viewers voting on the winner.

 Solo singers were also sought to join a group to represent the United Kingdom on the American glee club circuit.

 The show was hosted by Emma Bunton, and judged by Anastacia, Duncan James, Tamzin Outhwaite and Charles "Chucky" Klapow.

 The programme was shown in simulcast on Irish TV channel 3e and repeated a week later on parent channel TV3 Ireland.

 The show was not renewed for a second series due to low ratings.
</item 6>

<item 7>
Graph query row:
person: Grantley Jonathan Nicholas
nationality: Welsh
occupation: ['musician', 'lead singer', 'lead guitarist']
</item 7>

<item 8>
Dan Clark
Daniel Gregory Clark (born 3 July 1976 in Bromley, London) is an English actor, comedian, writer, director, and singer.

 He is best known for playing Don Danbury on the BBC Three sitcom "How Not to Live Your Life", which he also wrote, co-produced, and sometimes directed.

 He has been a regular on the British comedy scene as both a sketch and stand-up comedian.
</item 8>

<item 9>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 9>

<item 10>
Nicholas Hytner
Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.

 He was previously the Artistic Director of London's National Theatre.

 His major successes while director include "Miss Saigon", "The History Boys" and "One Man, Two Guvnors".
</item 10>
=== END ITEM id=3bbf4ca044f77159 ===

=== ITEM id=dc428db7d992719c ===
Gold sentences and facts:
Sentence 0:  U2 did not play the song live until 2 July 2009, when they performed it on the second concert of their U2 360° Tour in Barcelona.
  Fact 0.0: U2 did not play the song live until 2 July 2009.
  Fact 0.1: U2 performed the song on 2 July 2009 on the second concert of their U2 360° Tour in Barcelona.
Sentence 1: The U2 360° Tour was a worldwide concert tour by rock band U2.
  Fact 1.0: The U2 360° Tour was a worldwide concert tour.
  Fact 1.1: The U2 360° Tour was by rock band U2.
Sentence 2:  The concerts featured the band playing "in the round" on a circular stage, allowing the audience to surround them on all sides.
  Fact 2.0: The U2 360° Tour concerts featured the band playing 'in the round'.
  Fact 2.1: The U2 360° Tour concerts used a circular stage.
  Fact 2.2: The circular stage allowed the audience to surround the band on all sides.

Retrieved items:
<item 1>
The Year of the Voyager
The Year Of The Voyager is a double DVD/CD released by Nevermore.

 It was released in Europe on October 20, 2008 and in North America on November 25, 2008 via Century Media.

 The set covers the "This Godless Endeavor" touring cycle, starting with live footage from the U.S. Gigantour 2005, the Metal Mania festival 2006 in Poland, the
Wacken Open Air festival in 2006 in Germany and the main DVD show recorded at the Zeche in Bochum, Germany.

 Bonus material includes two songs from Century Media USA 10th Anniversary Party 2001, all promo videos and an interview with singer Warrel Dane which was
recorded at the Roax Film Studios in Berlin in the spring of 2008.
</item 1>

<item 2>
Ceremonials Tour
The Ceremonials Tour was the second concert tour by English indie rock band Florence and the Machine.

 The tour included performances at Music festivals because it is lead singer Florence Welch's favourite way to perform live.

 Welch had originally planned to spend over one year touring for "Ceremonials" (their previous tour lasted almost three and a half years) before announcing that
the December 2012 dates would be the final performances of the tour.

 On 25 February 2013 it was announced that Florence and the Machine would be playing at the Coke Live Music Festival in Poland on 10 August 2013, and on 26
March 2013 it was announced that the band would be playing at Chime for Change's "The Sound of Change Live" concert at London's Twickenham Stadium alongside
Beyoncé, Ellie Goulding and Haim, amongst others.
</item 2>

<item 3>
Psycho Circus World Tour
This ultimately happened only at the first concert at Dodger Stadium.

 Peter Criss later said that it didn't work out because the circus performers wanted equal billing and that some had even wanted to use KISS' backstage dressing
room.

 "2,000 Man" was played to bring in the new Millennium at the 1999/2000 New Year's Eve show at Vancouver, advertised at the time as being recorded for Alive IV.

 The Vancouver show was also notable as being the first time the original members had played non-original band era material live in concert – "I Love It Loud",
"Lick It Up" and "Heaven's on Fire" were added to the setlist and subsequently played on the Farewell Tour.

 "Forever" was listed on concert setlists at the Vancouver show but was not played.

 It was thought at the time it may have been intended as a Paul Stanley solo version prior to "Black Diamond".

 One notable show on the tour was the March 12 Bremen, Germany, show.

 After the opening song, Paul Stanley announced that the local fire marshall had banned Kiss from using any pyrotechnics during the show.
</item 3>

<item 4>
U2 360° Tour
The U2 360° Tour was a worldwide concert tour by rock band U2.

 Launched in support of the group's 2009 album "No Line on the Horizon", the tour visited stadiums from 2009 through 2011.

 The concerts featured the band playing "in the round" on a circular stage, allowing the audience to surround them on all sides.

 To accommodate the stage configuration, a large four-legged structure nicknamed "The Claw" was built above the stage, with the sound system and a cylindrical,
expanding video screen on top of it.

 At 164 ft tall, it was the largest stage ever constructed.

 U2 claimed that the tour would be "the first time a band has toured in stadiums with such a unique and original structure."
</item 4>

<item 5>
Rock &amp; Roll Over Tour
Rock and Roll Over Tour is a concert tour by the American hard rock group Kiss.

 It began November 24, 1976 (shortly after the release of the Rock and Roll Over album) and ended April 4, 1977.

 It was on this tour where the band performed for the first time at Madison Square Garden on 2/18/77 - a venue all four members of the band had long dreamed of
playing (February 18, 1977).

 The band also performed in Japan for the first time on this tour.

 They played Budokan Hall four nights in a row breaking the previous concert attendance record set by The Beatles.

 On December 12, 1976 Ace Frehley suffered a severe electrical shock on stage which delayed the concert.

 The delay lasted about fifteen minutes and then resumed; Ace was able to finish the show despite having lost feeling in one of his hands.

 Being inspired by the event, Ace wrote the song "Shock Me" which appeared on Kiss' next album release Love Gun (1977).

 It is the first Kiss song in which Ace sings lead vocals.
</item 5>

<item 6>
Because We Can (concert tour)
Because We Can was a concert tour by American rock band, Bon Jovi.

 The tour was in support of the band's twelfth studio album "What About Now".

 The tour was named after the lead single from "What About Now".

 All five major continents were visited during the 2013 tour.

 The tour saw the band travel to Wales for the first time since their 2001 One Wild Night Tour and was also the first time since their 1995 These Days Tour that
the band returned to Africa for two dates in South Africa.

 Furthermore, this tour was the first time in the band's 30-year history that they visited Bulgaria and Poland.

 The tour ranked 1st on Pollstar's annual "Top 100 Mid Year Worldwide Tours".

 It earned $142.1 million from 60 shows.

 At the end of 2013, the tour placed 1st on Pollstar's "Top 100 Worldwide Tours", grossing $259.5 million from 102 shows.
</item 6>

<item 7>
Psycho Circus World Tour
They used a translator on stage to let the crowd understand exactly what Stanley was saying.

 At the end of the performance, the band ignited all of the pyrotechnics at once; as a result, they were banned from performing in Bremen.

 Ticket sales for this tour were notably slower than the previous Reunion Tour, with many of the smaller market shows underselling and a second North American
leg for the summer of 1999 cancelled all together, the band ultimatly decided on embarking on a Farewell tour in the new millennium.
</item 7>

<item 8>
Escape Tour
The Escape Tour was a concert tour by the American rock band Journey.

 It was one of Journey's most successful concert tours, in support of their first number one album, "Escape".

 The tour included six consecutive sold out dates at the Pine Knob Theatre in Detroit, and four straight sold out shows at the Los Angeles Forum, and Chicago's
Rosemont Horizon.

 Journey also made an appearance on July 2, 1982 at the Rose Bowl in Pasadena, California with Blue Öyster Cult, Triumph and Aldo Nova.

 The 8 ⁄ -month tour took Journey through Japan and North America.

 Point Blank, Billy Squier, the Greg Kihn Band and Loverboy were the opening acts during the tour.
</item 8>

<item 9>
World on Fire World Tour
The World on Fire World Tour is the third concert tour by lead guitarist Slash as a solo artist, which started in July 2014 and was scheduled to resume in late
2015, in support of Slash's third solo album "World on Fire".

 The tour features the same backing band that performed with Slash during his two last tours, billed as "Slash featuring Myles Kennedy and The Conspirators",
featuring Myles Kennedy handling lead vocal, bassist Todd Kerns, drummer Brent Fitz and rhythm guitarist Frank Sidoris.
</item 9>

<item 10>
Psycho Circus World Tour
Psycho Circus World Tour was a Kiss concert tour in 1998–2000.

 It was the first concert tour in history to have 3-D visual effects.

 The Smashing Pumpkins opened at the Dodger Stadium show only, in costume as The Beatles for the Halloween night performance.

 The Dodger Stadium show was streamed live on the internet as well as a radio broadcast.

 Two songs, "Psycho Circus" and "Shout It Out Loud", were screened live on Fox television as part of the "Kiss Live: The Ultimate Halloween Party" special.

 The vast majority of songs in the setlist were played on the previous Alive/Worldwide concert dates, leading to some frustration from fans expecting the return
of classic songs not played on the previous tour.

 Peter Criss was quoted in Metal Edge magazine at the time as wanting to add "Parasite" to the setlist.

 The tour was initially hyped as having circus-style acts as pre-show entertainment.

 This ultimately happened only at the first concert at Dodger Stadium.
</item 10>
=== END ITEM id=dc428db7d992719c ===

=== ITEM id=393b2c052f4f8540 ===
Gold sentences and facts:
Sentence 0: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham and province of the Catholic Church in Great Britain and is dedicated to Saint Chad of Mercia.
  Fact 0.0: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham.
  Fact 0.1: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the province of the Catholic Church in Great Britain.
  Fact 0.2: The Metropolitan Cathedral Church and Basilica of Saint Chad is dedicated to Saint Chad of Mercia.
Sentence 1:  Built by Augustus Welby Pugin and substantially complete by 1841, St Chad's is one of the first four Catholic churches that were constructed after the English Reformation and raised to cathedral status in 1852.
  Fact 1.0: St Chad's was built by Augustus Welby Pugin.
  Fact 1.1: St Chad's was substantially complete by 1841.
  Fact 1.2: St Chad's is one of the first four Catholic churches constructed after the English Reformation.
  Fact 1.3: St Chad's was raised to cathedral status in 1852.
Sentence 2: Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.
  Fact 2.0: Chelmsford Cathedral is in the city of Chelmsford, Essex, England.
  Fact 2.1: Chelmsford Cathedral is dedicated to St Mary the Virgin.
  Fact 2.2: Chelmsford Cathedral is dedicated to St Peter.
  Fact 2.3: Chelmsford Cathedral is dedicated to St Cedd.
Sentence 3:  It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
  Fact 3.0: Chelmsford Cathedral became a cathedral when the Anglican Diocese of Chelmsford was created in 1914.
  Fact 3.1: Chelmsford Cathedral is the seat of the Bishop of Chelmsford.

Retrieved items:
<item 1>
St Chad's College, Durham
St Chad's College is a recognised (independent) college of Durham University in England, founded in 1904 as an Anglican hall for the training of Church of
England clergy.

 The main part of the college is located on the Bailey, occupying nine historic buildings at the east end of Durham Cathedral.

 It neighbours Hatfield College to its north, while St John's College and St Cuthbert's Society are to its south.

 The college is named after St Chad of Mercia, a 7th-century bishop.
</item 1>

<item 2>
Chelmsford Cathedral
Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.

 It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
</item 2>

<item 3>
Brentwood Cathedral
The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

 It is the seat of the Diocese of Brentwood.
</item 3>

<item 4>
Dean of Chelmsford
The Dean of Chelmsford is the head ("primus inter pares" – first among equals) and chair of the Cathedral Chapter, the governing body of Chelmsford Cathedral,
the Cathedral Church of St Mary the Virgin, St Peter and St Cedd.

 Before 2000 the post was designated as a provost, which was then the equivalent of a dean at most English cathedrals.

 The cathedral is the mother church of the Diocese of Chelmsford and seat of the Bishop of Chelmsford.

 The Dean of Chelmsford is also responsible for the Chapel of St Peter-on-the-Wall at Bradwell-on-Sea, founded by St Cedd, among the oldest church buildings in
regular use in England.
</item 4>

<item 5>
Guildford Cathedral
The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.

 The Lord Onslow donated the land on which the Cathedral was built with a covenant for this purpose.

 Designed by Sir Edward Maufe and built between 1936 and 1961, it is the seat of the Diocese of Guildford.
</item 5>

<item 6>
Lincoln Cathedral
Lincoln Cathedral or the Cathedral Church of the Blessed Virgin Mary of Lincoln, and sometimes St. Mary's Cathedral in Lincoln, England is the seat of the
Anglican bishop.

 Building commenced in 1088 and continued in several phases throughout the medieval period.

 It was the tallest building in the world for 238 years (1311–1549), and the first building to hold that title after the Great Pyramid of Giza.

 The central spire collapsed in 1549 and was not rebuilt.

 The cathedral is the third largest in Britain (in floor area) after St Paul's and York Minster, being 484 by .

 It is highly regarded by architectural scholars; the eminent Victorian writer John Ruskin declared: "I have always held... that the cathedral of Lincoln is out
and out the most precious piece of architecture in the British Isles and roughly speaking worth any two other cathedrals we have."
</item 6>

<item 7>
Bishop's House, Birmingham
The Bishop's House in Birmingham, England was designed by Augustus Pugin as the residence of Thomas Walsh, the first Roman Catholic Bishop of Birmingham.

 It was situated opposite St Chad's Cathedral, on the corner of Bath Street and Weaman Street in Birmingham City Centre.
</item 7>

<item 8>
St Chad's Cathedral, Birmingham
The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham and province of the Catholic Church in Great
Britain and is dedicated to Saint Chad of Mercia.

 Built by Augustus Welby Pugin and substantially complete by 1841, St Chad's is one of the first four Catholic churches that were constructed after the English
Reformation and raised to cathedral status in 1852.

 It is one of only four minor basilicas in England (the others being Downside Abbey, the National Shrine of Our Lady at Walsingham and Corpus Christi Priory,
this last now disused).

 St Chad's is a Grade II* listed building.

 The cathedral is located in a public greenspace near St Chad's Queensway, in central Birmingham.

 s of 2014 the Archbishop was Bernard Longley and the Dean Canon Gerry Breen.
</item 8>

<item 9>
Lichfield Cathedral
Lichfield Cathedral is situated in Lichfield, Staffordshire, England.

 It is the only medieval English cathedral with three spires.

 The Diocese of Lichfield covers all of Staffordshire, much of Shropshire and part of the Black Country and West Midlands.

 The 99th and current Bishop of Lichfield is Michael Ipgrave who was appointed on 10 June 2016.
</item 9>

<item 10>
St Philip's Cathedral, Birmingham
The Cathedral Church of Saint Philip is the Church of England cathedral and the seat of the Bishop of Birmingham.

 Built as a parish church and consecrated in 1715, St Philip's became the cathedral of the newly formed Diocese of Birmingham in 1905.

 St Philip's was built in the early 18th century in the Baroque style by Thomas Archer and is located on Colmore Row, Birmingham, England.

 The cathedral is a Grade I listed building.

 St Philip's is the third smallest cathedral in England after Derby and Chelmsford.
</item 10>
=== END ITEM id=393b2c052f4f8540 ===

=== ITEM id=b095ca1c0145d938 ===
Gold sentences and facts:
Sentence 0:  During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas" (1992), and "Red Rock West" (1993).
  Fact 0.0: During his early career, Cage starred in a variety of films.
  Fact 0.1: Cage starred in Valley Girl, released in 1983.
  Fact 0.2: Cage starred in Racing with the Moon, released in 1984.
  Fact 0.3: Cage starred in Birdy, released in 1984.
  Fact 0.4: Cage starred in Peggy Sue Got Married, released in 1986.
  Fact 0.5: Cage starred in Raising Arizona, released in 1987.
  Fact 0.6: Cage starred in Moonstruck, released in 1987.
  Fact 0.7: Cage starred in Vampire's Kiss, released in 1989.
  Fact 0.8: Cage starred in Wild at Heart, released in 1990.
  Fact 0.9: Cage starred in Fire Birds, released in 1990.
  Fact 0.10: Cage starred in Honeymoon in Vegas, released in 1992.
  Fact 0.11: Cage starred in Red Rock West, released in 1993.
Sentence 1: Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.
  Fact 1.0: Moonstruck is a 1987 American film.
  Fact 1.1: Moonstruck is a romantic comedy film.
  Fact 1.2: Moonstruck was directed by Norman Jewison.
  Fact 1.3: Moonstruck was written by John Patrick Shanley.
Sentence 2:  It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother (Nicolas Cage).
  Fact 2.0: Moonstruck is about a widowed 37-year-old Italian-American woman played by Cher.
  Fact 2.1: The woman falls in love with her fiancé's estranged, hot-tempered younger brother.
  Fact 2.2: The fiancé is played by Danny Aiello.
  Fact 2.3: The younger brother is played by Nicolas Cage.

Retrieved items:
<item 1>
Cool and the Crazy
Cool and the Crazy is a 1994 American drama film written and directed by Ralph Bakshi and starring Jared Leto and Alicia Silverstone.

 The story revolves around an unhappily married couple in the late 1950s who both lead separate affairs.

 The film was Bakshi's first feature-length live-action film, being primarily known as a director of animated films which heavily utilize live-action sequences,
such as "Fritz the Cat", "Heavy Traffic", "Wizards", "American Pop" and "The Lord of the Rings".
</item 1>

<item 2>
Where Do You Go (Cher song)
"Where Do You Go" is a song written by Sonny Bono.

 It was released as the first single by Cher in the later quarter of 1965 for her second album "The Sonny Side of Cher".

 It fell short of the "Billboard" Hot 100's top 20, but still earned Cher a moderate success, by reaching the Top 40.

 It was followed by the U.S. #2 smash hit "Bang Bang (My Baby Shot Me Down)" which saw a release early the following year.

 It was a bigger hit in Canada, where it reached #5 on the singles chart.
</item 2>

<item 3>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 3>

<item 4>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 4>

<item 5>
Nicolas Cage
Nicolas Kim Coppola (born January 7, 1964), known professionally as Nicolas Cage, is an American actor, director and producer.

 During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got
Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas"
(1992), and "Red Rock West" (1993).
</item 5>

<item 6>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 6>

<item 7>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 7>

<item 8>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 8>

<item 9>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 9>

<item 10>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 10>
=== END ITEM id=b095ca1c0145d938 ===
