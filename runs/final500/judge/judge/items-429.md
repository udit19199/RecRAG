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

=== ITEM id=05212074beb1f982 ===
Gold sentences and facts:
Sentence 0:  During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas" (1992), and "Red Rock West" (1993).
  Fact 0.0: During his early career, Nicolas Cage starred in a variety of films.
  Fact 0.1: Nicolas Cage starred in Valley Girl (1983).
  Fact 0.2: Nicolas Cage starred in Racing with the Moon (1984).
  Fact 0.3: Nicolas Cage starred in Birdy (1984).
  Fact 0.4: Nicolas Cage starred in Peggy Sue Got Married (1986).
  Fact 0.5: Nicolas Cage starred in Raising Arizona (1987).
  Fact 0.6: Nicolas Cage starred in Moonstruck (1987).
  Fact 0.7: Nicolas Cage starred in Vampire's Kiss (1989).
  Fact 0.8: Nicolas Cage starred in Wild at Heart (1990).
  Fact 0.9: Nicolas Cage starred in Fire Birds (1990).
  Fact 0.10: Nicolas Cage starred in Honeymoon in Vegas (1992).
  Fact 0.11: Nicolas Cage starred in Red Rock West (1993).
Sentence 1:  It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother (Nicolas Cage).
  Fact 1.0: The film is about a widowed 37-year-old Italian-American woman.
  Fact 1.1: Cher plays the widowed 37-year-old Italian-American woman.
  Fact 1.2: The widowed woman falls in love with her fiancé's estranged, hot-tempered younger brother.
  Fact 1.3: The widowed woman's fiancé is played by Danny Aiello.
  Fact 1.4: The fiancé's younger brother is estranged.
  Fact 1.5: The fiancé's younger brother is hot-tempered.
  Fact 1.6: The fiancé's younger brother is played by Nicolas Cage.

Retrieved items:
<item 1>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 1>

<item 2>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 2>

<item 3>
3:10 to Yuma (2007 film)
3:10 to Yuma is a 2007 American Western film directed by James Mangold and produced by Cathy Konrad, and starring Russell Crowe and Christian Bale in the lead
roles, with supporting performances by Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman.

 It is about a drought-impoverished rancher (Bale) who takes on the dangerous job of taking a notorious outlaw (Crowe) to justice.

 It is a remake of the , making it the second adaptation of Elmore Leonard's short story "Three-Ten to Yuma".

 Filming took place in various locations in New Mexico.

 "3:10 to Yuma" opened September 7, 2007, in the United States and received positive reviews from critics.
</item 3>

<item 4>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 4>

<item 5>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 5>

<item 6>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 6>

<item 7>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 7>

<item 8>
Nicolas Cage
Nicolas Kim Coppola (born January 7, 1964), known professionally as Nicolas Cage, is an American actor, director and producer.

 During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got
Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas"
(1992), and "Red Rock West" (1993).
</item 8>

<item 9>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 9>

<item 10>
A Star Is Born (1976 film)
A Star Is Born is a 1976 American musical drama film telling the story of a young woman, played by Barbra Streisand, who enters show business, and meets and
falls in love with an established male star, played by Kris Kristofferson, only to find her career ascending while his goes into decline.

 It is a remake of two earlier versions – the 1937 version was a drama starring Janet Gaynor and Fredric March, and the 1954 version was a musical starring Judy
Garland and James Mason.

 It will be remade for a third time in 2018 starring Stefani Germanotta and Bradley Cooper.
</item 10>
=== END ITEM id=05212074beb1f982 ===

=== ITEM id=653b282af347bac3 ===
Gold sentences and facts:
Sentence 0: Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.
  Fact 0.0: Bertalan Farkas was born on August 2, 1949.
  Fact 0.1: Bertalan Farkas is the first Hungarian cosmonaut.
  Fact 0.2: Bertalan Farkas is the first Esperantist in space.
Sentence 1: Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
  Fact 1.0: Hans Wilhelm Schlegel was born in Überlingen on 3 August 1951.
  Fact 1.1: Hans Wilhelm Schlegel is a German physicist.
  Fact 1.2: Hans Wilhelm Schlegel is an ESA astronaut.
  Fact 1.3: Hans Wilhelm Schlegel is a veteran of two NASA Space Shuttle missions.

Retrieved items:
<item 1>
Hans Aschenborn
Hans Anton Aschenborn (1 February 1888 – 10 April 1931) was a renowned animal painter of African wildlife.

 He worked both in Germany and in southern Africa.

 His work is featured in the older German Thieme-Becker or Saur art encyclopedia.

 A Master of Arts thesis by Karin Skawran concerning the graphic works of Hans Anton Aschenborn was published in the South African art and culture periodical,
"Lantern" in 1965.)

 In 1963 the University of Pretoria published a book about Aschenborn as an artist entitled, "Hans Anton Aschenborn : Mens en Kunstenaar".

 In 1970 another one followed by the Pretoria Art Museum (South Africa).

 Other publications on Aschenborn feature his etchings and linocuts.
</item 1>

<item 2>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 2>

<item 3>
Hans von Hentig
Hans von Hentig (9 June 1887 in Berlin – 6 July 1974 in Bad Tölz) was a German criminal psychologist and politician.He was the second son of lawyer Otto von
Hentig (1852–1934) .

His older brother was later diplomat Werner Otto von Hentig.

 Otto von Hentig was one of the leading lawyers in Berlin.

 Hans von Hentig was instrumental in the setting up of a short-lived Bavarian Soviet Republic in 1919.

 During the 1920s he was a prominent exponent of National Bolshevism.

 He emigrated to United States in 1935.

 Hans von Hentig worked on Yale University.
</item 3>

<item 4>
Karl Anselm, 4th Prince of Thurn and Taxis
Karl Anselm, 4th Prince of Thurn and Taxis, full German name: "Karl Anselm Fürst von Thurn und Taxis" (2 June 1733, Frankfurt am Main, Free Imperial City of
Frankfurt, Holy Roman Empire – 13 November 1805, Winzer bei Regensburg, Electorate of Bavaria, Holy Roman Empire) was the fourth Prince of Thurn and Taxis,
Postmaster General of the Imperial Reichspost, and Head of the Princely House of Thurn and Taxis from 17 March 1773 until his death on 13 November 1805.

 Karl Anselm served as "Prinzipalkommissar" at the Perpetual Imperial Diet in Regensburg for Joseph II, Holy Roman Emperor and Francis II, Holy Roman Emperor
from 1773 to 1797.
</item 4>

<item 5>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 5>

<item 6>
Gottlieb Muffat
Gottlieb Muffat (April 1690 – 9 December 1770), son of Georg Muffat, served as "Hofscholar" under Johann Fux in Vienna from 1711 and was appointed to the
position of third court organist at the "Hofkapelle" in 1717.

 He acquired additional duties over time including the instruction of members of the Imperial family, among them the future Empress Maria Theresa.

 He was promoted to second organist in 1729 and first organist upon the accession of Maria Theresa to the throne in 1741.

 He retired from official duties at the court in 1763.
</item 6>

<item 7>
Karl Wilhelm Friedrich Schlegel
Karl Wilhelm Friedrich (after 1814: von) Schlegel (10 March 1772 – 12 January 1829), usually cited as Friedrich Schlegel, was a German poet, literary critic,
philosopher, philologist and Indologist.

 With his older brother, August Wilhelm Schlegel, he was one of the main figures of the Jena romantics.

 He was a zealous promoter of the Romantic movement and inspired Samuel Taylor Coleridge, Adam Mickiewicz and Kazimierz Brodziński.

 Schlegel was a pioneer in Indo-European studies, comparative linguistics, morphological typology, and was the first to notice what became known as Grimm's law.
</item 7>

<item 8>
Richard Genée
Franz Friedrich Richard Genée (February 7, 1823 – June 15, 1895) was a Prussian born Austrian librettist, playwright, and composer.

 He is most famous for the libretto of "Die Fledermaus", Johann Strauss II's most famous operetta.

 He co-wrote the libretto without having met top-billed librettist Karl Haffner, who constructed the new story based on a play by Henri Meilhac and Ludovic
Halévy, which was considered too shocking to perform outside Paris.

 Genée, however, wrote the operetta's actual text and drew nothing from Haffner beyond the names of the characters.
</item 8>

<item 9>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 9>

<item 10>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 10>
=== END ITEM id=653b282af347bac3 ===

=== ITEM id=13dc6291e7563f4d ===
Gold sentences and facts:
Sentence 0: Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.
  Fact 0.0: Semyon Aronovich Gershgorin was born on August 24, 1901.
  Fact 0.1: Semyon Aronovich Gershgorin died on May 30, 1933.
  Fact 0.2: Semyon Aronovich Gershgorin was Soviet.
  Fact 0.3: Semyon Aronovich Gershgorin was born in Pruzhany, Belarus, Russian Empire.
  Fact 0.4: Semyon Aronovich Gershgorin was a mathematician.
Sentence 1:  He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after him.
  Fact 1.0: Semyon Aronovich Gershgorin was a mathematician.
  Fact 1.1: Semyon Aronovich Gershgorin was noted for his work on the decidability of various algebraic groups.
  Fact 1.2: Malcev algebras are generalisations of Lie algebras.
  Fact 1.3: Malcev algebras are named after Semyon Aronovich Gershgorin.

Retrieved items:
<item 1>
Nicolai V. Krylov
Bellman equations), which were examined with stochastic methods.

 This led to the Evans-Krylov theory, for which he received with Lawrence C. Evans in 2004 the Leroy P. Steele Prize of the American Mathematical Society (for
work done simultaneously and independently by both Krylov and Evans).

 They proved the second order differentiability (Hölder continuity of the second derivative) of the solutions of convex, completely nonlinear, second order
elliptical partial differential equations and thus the existence of "classical solutions" (Theorem of Evans-Krylov).

 He was in 1978 at Helsinki and in 1986 at Berkeley an Invited Speaker for the ICM.

 He received the Humboldt Research Award in 2001.

 In 1993 he was elected a member of the American Academy of Arts and Sciences (1993).

 He should not be confused with the mathematician Nikolay M. Krylov.
</item 1>

<item 2>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 2>

<item 3>
Nikolay Mitrofanovich Krylov
Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11,
1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical
physics.
</item 3>

<item 4>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 4>

<item 5>
Graph query row:
person: Semyon Aranovich Gershgorin
occupation: ['mathematician', 'student', 'Professor']
relationship: None
discipline: None
</item 5>

<item 6>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 6>

<item 7>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 7>

<item 8>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 8>

<item 9>
Graph query row:
person: Anatoly Maltsev
occupation: mathematician
relationship: None
discipline: None
</item 9>

<item 10>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 10>
=== END ITEM id=13dc6291e7563f4d ===

=== ITEM id=f62c4b0f353cd06c ===
Gold sentences and facts:
Sentence 0:  She was born in Guangzhou, China.
  Fact 0.0: The woman referred to as "she" was born in Guangzhou, China.
Sentence 1:  These videos are an overview and recap of each year's viral videos, events, memes, and music.
  Fact 1.0: The videos being described are an overview of each year's viral videos, events, memes, and music.
  Fact 1.1: The videos being described are a recap of each year's viral videos, events, memes, and music.

Retrieved items:
<item 1>
Lilly Singh
Lilly Singh (born 26 September 1988) is a Canadian YouTube personality, vlogger, comedian, writer, and actress.

 She is better known by her YouTube username IISuperwomanII.

 Since beginning her channel in October 2010, her videos have received over 2 billion views, and her channel has accumulated over 12 million subscribers.

 In 2016, she was ranked 3rd on the "Forbes" list of the world's highest paid YouTube stars (behind Roman Atwood and PewDiePie), earning a reported $7.5 million
in 2016.

 Singh has featured in the annual YouTube Rewind every year since 2014.

 She ranked 1st on 2017 Forbes Top Influencers List in the entertainment category.
</item 1>

<item 2>
XVideos
XVideos is the world's largest pornographic video sharing and viewing website.

 It's registered to the Polish company, WGCZ Holding.

 Alexa ranks it as the 50th most visited website in the world.
</item 2>

<item 3>
Wayne Goss (make-up artist)
Wayne Alan Goss (born 4 March 1978) is an English makeup artist, YouTube personality and entrepreneur.

 Best known for his "Wayne Goss" channel on YouTube, he is the creator and owner of the self-eponymous makeup brush line launched in September 2013.

 As of November 2016, Goss' main YouTube channel had over 2.7 million subscribers and had had 344 million views.
</item 3>

<item 4>
Issa Rae
Jo-Issa "Issa" Rae Diop (born January 12, 1985) is an American actress, writer, director, producer and web series creator.

 She is best known as the creator of the YouTube web series "Awkward Black Girl".

 Since the premiere of "Awkward Black Girl", Rae has developed her own YouTube platform where she features various content created by people of color.

 Rae's shows have garnered over 20 million views and over 260,000 subscribers on YouTube.
</item 4>

<item 5>
CVX Live
CVX Live (Creator Viewer Experience Live) is an event designed to bring online media creators and their fans together.

 It was held for the first time in Orem, Utah, on August 7 to 8, 2015 Originally created by the team behind the "Bored Shorts TV" YouTube channel, the
convention is Utah's first YouTube event, gathering thousands of online YouTube creators and fans together in one place.

 One of the creators, Brett Robers, described it as “...a very unique opportunity to bring YouTube to life.”
</item 5>

<item 6>
YouTube Rewind
YouTube Rewind is a video series produced and created by YouTube and Portal A Interactive.

 These videos are an overview and recap of each year's viral videos, events, memes, and music.

 Each year, the number of YouTube celebrities featured in the video, as well as the presentation of the series, have increased.

 The latest episode of "YouTube Rewind" was released on December 8, 2016.
</item 6>

<item 7>
Wengie
Wendy Ayche (born January 9 1986) known professionally as Wengie, is an Australian YouTube personality and vlogger.

 She was born in Guangzhou, China.

 She got the nickname "Wengie" at a dance class when two people gave her a nickname inspired by her Chinese name, Wén Jié.

 Since starting her channel on February 11 2013, her videos have received over 281 million views, and her channel has accumulated over 11 followers In the
January 2016 edition of "ElleGirl Japan", she was nominated as a channel to watch.

 In August 2016, she was ranked the 5th Fastest growing channel in the world.

 Ayche was featured in the annual YouTube Rewind in 2016 and was also singled out by Google as one of the top beauty creators in the Asia region.

 Ayche reached 5 million YouTube subscribers in January 2017, and her channel is currently in the 12th most subscribed How-To & Style Channel on YouTube.

 She is also currently a board member of the Internet Creators Guild, a non-profit focused on providing the protection, representation and guidance to online
creators.
</item 7>

<item 8>
Wengie
Her YouTube channel also recently got awarded with "Best Channel" as well as "Overall Winner" for the Australian Online Video Awards.
</item 8>

<item 9>
Kevin Nalty
Kevin 'Nalts' Nalty (born May 12, 1969) is a YouTube comedian and partner based in Doylestown, Pennsylvania better known under his YouTube username Nalts.

 Nalts began on YouTube as one of the top-20 most-viewed comedy channels, and collaborates with many of today's top YouTube personalities.

 He has more than 1,000 videos which, as of April 18, 2016, have been viewed more than 297 million times on YouTube alone.

 He has been ranked as one of YouTube's Most Subscribed users.

 He is the author of "Beyond Viral: How to Attract Customers, Promote Your Brand, and Make Money with Online Video (Wiley & Sons, 2010).
</item 9>

<item 10>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 10>
=== END ITEM id=f62c4b0f353cd06c ===

=== ITEM id=c53b3c8a92fa6e64 ===
Gold sentences and facts:
Sentence 0: Kresy Wschodnie or Kresy (] , Eastern Borderlands, or Borderlands) was a region of the Second Polish Republic during the interwar period constituting nearly half of the territory of the state; where the ethnic Poles, being the largest group, were roughly equal in their number to the size of the national minorities (with notable exceptions).
  Fact 0.0: Kresy Wschodnie, also called Kresy, Eastern Borderlands, or Borderlands, was a region of the Second Polish Republic during the interwar period.
  Fact 0.1: Kresy constituted nearly half of the territory of the Second Polish Republic.
  Fact 0.2: In Kresy, ethnic Poles were the largest group.
  Fact 0.3: In Kresy, ethnic Poles were roughly equal in number to the size of the national minorities.
  Fact 0.4: The rough equality of ethnic Poles with the national minorities in Kresy had notable exceptions.
Sentence 1:  It was drawn for the first time by the Supreme War Council as the demarcation line between the newly emerging states, the Second Polish Republic, and the Soviet Union.
  Fact 1.0: Kresy was drawn for the first time by the Supreme War Council.
  Fact 1.1: Kresy was drawn as the demarcation line between the newly emerging states, the Second Polish Republic and the Soviet Union.

Retrieved items:
<item 1>
Kresy
Kresy Wschodnie or Kresy (] , Eastern Borderlands, or Borderlands) was a region of the Second Polish Republic during the interwar period constituting nearly
half of the territory of the state; where the ethnic Poles, being the largest group, were roughly equal in their number to the size of the national minorities
(with notable exceptions).

 Administratively, the territory of Kresy was composed of voivodeships of Lwów, Nowogródek, Polesie, Stanisławów, Tarnopol, Wilno, Wołyń, and the Białystok.

 Today, these territories are divided between Western Ukraine, Western Belarus, and south-eastern Lithuania, with such major cities as Lviv, Vilnius, and Grodno
no longer in Poland.

 In the Second Polish Republic the term "Kresy" roughly equated with the lands beyond the so-called Curzon Line, which was suggested after World War I in
December 1919 by the British Foreign Office as the eastern border of the re-emerging sovereign Republic following the century of partitions.
</item 1>

<item 2>
Battle of Berestechko
The Battle of Berestechko (Polish: "Bitwa pod Beresteczkiem" ; Ukrainian: Берестецька битва, Битва під Берестечком ) was fought between the Ukrainian Cossacks,
led by Hetman Bohdan Khmelnytsky, aided by their Crimean Tatar allies, and a Polish army under King John II Casimir.

 It was a battle of a Cossack rebellion in Ukraine that took place in the years 1648–1657 after the expiration of a two-year truce.

 Fought from 28 to 30 June 1651, the battle took place in the province of Volhynia, on the hilly plain south of the Styr River.

 The Polish camp was on the river opposite Berestechko and faced south, towards the Cossack army about two kilometers away, whose right flank was against the
River Pliashivka (Pliashova) and the Tatar army on their left flank.

 It was probably the largest European land battle of the 17th century.
</item 2>

<item 3>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 3>

<item 4>
Kresy
In September 1939, after the Soviet Union joined Nazi Germany in their attack on Poland in accordance with the Molotov–Ribbentrop Pact, the territories were
incorporated into Soviet Ukraine, Belarus and Lithuania in the atmosphere of terror.
</item 4>

<item 5>
Anti-Nebraska movement
The Anti-Nebraska movement was a political alignment in the United States formed in opposition to the Kansas-Nebraska Act of 1854 and to its repeal of the
Missouri Compromise provision forbidding slavery in U.S. territories north of latitude 36° 30' N.

 (At the time, the name "Nebraska" could loosely refer to areas west of the Missouri River.)
</item 5>

<item 6>
Lithuanian Education Society Rytas
The Lithuanian Education Society Rytas (Lithuanian: "Lietuvių švietimo draugija „Rytas“" ; "rytas" means "morning" or "dawn") was a Roman Catholic society
fostering education in the Lithuanian language mostly in the Vilnius Region, then fiercely contested between Lithuania and the Second Polish Republic, now split
between Lithuanian and Belarus.

 Established in 1913, the society maintained some 100 primary schools (mostly one-room schools), 50 evening classes, and Teacher's Seminary in Vilnius (Wilno,
Vilna) in 1927.

 Due to political tensions between Poland and Lithuania as well as wider Polonization policies, Rytas faced increasing difficulties and restrictions in
maintaining its schools.

 Similar situation existed with Polish schools in Lithuania (see Lithuanization).

 The situation continued to worsen as both sides increased restrictions in retribution.

 As schools were closed, Rytas shifted its focus to maintaining community reading rooms.
</item 6>

<item 7>
Curzon Line
The history of the Curzon Line, with minor variations, goes back to the period following World War I.

 It was drawn for the first time by the Supreme War Council as the demarcation line between the newly emerging states, the Second Polish Republic, and the
Soviet Union.

 The proposal was put forward by British Foreign Secretary George Curzon, to serve as a diplomatic basis for the future border agreement, and in that form, it
never materialized because the war went on.
</item 7>

<item 8>
History of the New Jersey State Constitution
Originally, the state of New Jersey was a single British colony, the Province of New Jersey.

 After the English Civil War, Charles II assigned New Jersey as a proprietary colony to be held jointly by Sir George Carteret and John Berkeley, 1st Baron
Berkeley of Stratton.

 Eventually, the collection of land fees, or quit-rents, from colonists proved inadequate for colonial profitability.

 Sir George Carteret sold his share of the colony to the Quakers in 1673.

 Following the sale, the land was divided into East and West Jersey.

 In 1681, West Jersey adopted a constitution.

 In 1683, East Jersey adopted one as well.

 In 1702, the colonies were united again under Anne, Queen of Great Britain, and adopted a constitution in 1776.
</item 8>

<item 9>
Tripartite Indenture
The Tripartite Indenture was an agreement made in February 1405 between Owain Glyndŵr, Edmund Mortimer, and Henry Percy, 1st Earl of Northumberland, agreeing to
divide England and Wales up between them at the expense of Henry IV.

 Glyndŵr was to be given Wales, and a substantial part of the west of England, including the English portions of the Welsh Marches.

 Northumberland was to have received the north, as well as Northamptonshire, Norfolk, Warwickshire, and Leicestershire.

 The Mortimers were to have received the rest of southern England.
</item 9>

<item 10>
Register Cliff
Register Cliff is a sandstone cliff and featured key navigational landmark prominently listed in the 19th century guidebooks about the Oregon Trail, and a place
where many emigrants chiseled the names of their families on the soft stones of the cliff it was one of the key checkpoint landmarks for parties heading west
along the Platte River valley west of Fort John, Wyoming which allowed travelers to verify they were on the correct path up to South Pass and not moving into
impassable mountain terrainsgeographically, it is on the eastern ascent of the Continental divide leading upward out of the great plains in the east of the U.S.
state of Wyoming.

 It is notable as a historic landmark for 'registering' hundreds of emigrants on the Oregon Trail (thus also the other northern Emigrant Trails that split off
farther west such as the California Trail and Mormon Trail) who came to follow custom and inscribed their names on its rocks during the western migrations of
the 19th century.

 An estimated 500,000 emigrants used these trails from 1843–1869, with up to one-tenth dying along the way, usually due to disease.
</item 10>
=== END ITEM id=c53b3c8a92fa6e64 ===

=== ITEM id=2a9d3dfea25604cc ===
Gold sentences and facts:
Sentence 0: The Straight Story is a 1999 internationally co-produced biographical road drama film directed by David Lynch.
  Fact 0.0: The Straight Story is a 1999 film.
  Fact 0.1: The Straight Story is internationally co-produced.
  Fact 0.2: The Straight Story is a biographical road drama film.
  Fact 0.3: The Straight Story was directed by David Lynch.
Sentence 1: Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.
  Fact 1.0: Frozen is a 2013 film.
  Fact 1.1: Frozen is American.
  Fact 1.2: Frozen is a 3D computer-animated film.
  Fact 1.3: Frozen is a musical fantasy film.
  Fact 1.4: Frozen was produced by Walt Disney Animation Studios.
  Fact 1.5: Frozen was released by Walt Disney Pictures.

Retrieved items:
<item 1>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 1>

<item 2>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 2>

<item 3>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 3>

<item 4>
A Star Is Born (1976 film)
A Star Is Born is a 1976 American musical drama film telling the story of a young woman, played by Barbra Streisand, who enters show business, and meets and
falls in love with an established male star, played by Kris Kristofferson, only to find her career ascending while his goes into decline.

 It is a remake of two earlier versions – the 1937 version was a drama starring Janet Gaynor and Fredric March, and the 1954 version was a musical starring Judy
Garland and James Mason.

 It will be remade for a third time in 2018 starring Stefani Germanotta and Bradley Cooper.
</item 4>

<item 5>
Fire and Ice (1983 film)
Fire and Ice is a 1983 American adult animated epic high fantasy adventure film directed by Ralph Bakshi.

 The film, a collaboration between Bakshi and Frank Frazetta, was distributed by 20th Century Fox, which also distributed Bakshi's 1977 release, "Wizards".

 The animated feature, based on characters Bakshi and Frazetta co-created, was made using the process of rotoscoping, in which scenes were shot in live action
and then traced onto animation cels.
</item 5>

<item 6>
The Straight Story
The Straight Story is a 1999 internationally co-produced biographical road drama film directed by David Lynch.

 The film was edited and produced by Mary Sweeney, Lynch's longtime partner and co-worker.

 She co-wrote the script with John E. Roach.

 The film is based on the true story of Alvin Straight's 1994 journey across Iowa and Wisconsin on a lawn mower.

 Alvin (Richard Farnsworth) is an elderly World War II veteran who lives with his daughter Rose (Sissy Spacek), a kind woman with an intellectual disability.

 When he hears that his estranged brother Lyle (Harry Dean Stanton) has suffered a stroke, Alvin makes up his mind to go visit him and hopefully make amends
before he dies.

 Because Alvin's legs and eyes are too impaired for him to receive a driving license, he hitches a trailer to his recently purchased thirty-year-old John Deere
110 Lawn Tractor, having a maximum speed of about 5 miles per hour, and sets off on the 240 mile journey from Laurens, Iowa to Mount Zion, Wisconsin.
</item 6>

<item 7>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 7>

<item 8>
Cool and the Crazy
Cool and the Crazy is a 1994 American drama film written and directed by Ralph Bakshi and starring Jared Leto and Alicia Silverstone.

 The story revolves around an unhappily married couple in the late 1950s who both lead separate affairs.

 The film was Bakshi's first feature-length live-action film, being primarily known as a director of animated films which heavily utilize live-action sequences,
such as "Fritz the Cat", "Heavy Traffic", "Wizards", "American Pop" and "The Lord of the Rings".
</item 8>

<item 9>
The Blizzard (1923 film)
The Blizzard is a 1923 Swedish drama film directed by Mauritz Stiller, starring Einar Hanson, Mary Johnson, Pauline Brunius and Hugo Björne.

 The film's original Swedish title is Gunnar Hedes saga, which means "The story of Gunnar Hede".

 The narrative revolves a student who tries to save his family's mansion which is facing bankruptcy.

 The film is loosely based on the Selma Lagerlöf novel "The Tale of a Manor".
</item 9>

<item 10>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 10>
=== END ITEM id=2a9d3dfea25604cc ===

=== ITEM id=204a5caecf9f858c ===
Gold sentences and facts:
Sentence 0: Jack and Jill is an American bimonthly magazine for children 7 to 12 years old that takes its title from the nursery rhyme of the same name.
  Fact 0.0: Jack and Jill is an American magazine for children.
  Fact 0.1: Jack and Jill is published bimonthly.
  Fact 0.2: Jack and Jill is for children 7 to 12 years old.
  Fact 0.3: The title of Jack and Jill comes from the nursery rhyme of the same name.
Sentence 1: Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.
  Fact 1.0: Paper, stylized as PAPER, is a New York City-based magazine.
  Fact 1.1: Paper is an independent magazine.
  Fact 1.2: Paper focuses on fashion.
  Fact 1.3: Paper focuses on popular culture.
  Fact 1.4: Paper focuses on nightlife.
  Fact 1.5: Paper focuses on music.
  Fact 1.6: Paper focuses on art.
  Fact 1.7: Paper focuses on film.

Retrieved items:
<item 1>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 1>

<item 2>
Key demographic
Certain radio formats (especially those dubbed "classic") and television outlets may target persons 35 to 64, especially since the late 2000s recession wiped
out many career opportunities for millennials, since the 35–64 demographic has much more disposable income.
</item 2>

<item 3>
Jack and Jill (magazine)
Jack and Jill is an American bimonthly magazine for children 7 to 12 years old that takes its title from the nursery rhyme of the same name.

 It features stories and educational activities.
</item 3>

<item 4>
Paper (magazine)
Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.

 Past cover models include Kim Kardashian, Demi Lovato, Katy Perry, Miley Cyrus, Prince, CL, and Jennifer Lopez.
</item 4>

<item 5>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 5>

<item 6>
Mighty Jill Off
Mighty Jill Off is a 2D platform independently developed freeware video game designed by Anna Anthropy, with art by James Harvey and music by Andrew Toups.

 It stars a submissive named Jill, who has a boot fetish and is forced to climb up a tower after her Queen kicks her down it as punishment.

 Jill does this by jumping and slowly descending over obstacles.

 Jill can be defeated in one hit by these obstacles, but will return to the last check point.

 The game serves as an homage to the 1986 arcade game "Mighty Bomb Jack".

 It had follow-ups, such as "Mighty Jill Off - Jill Off Harder Edition" and "Jill Off With One Hand".

 Jill made a cameo appearance in the 2010 video game "Super Meat Boy" as a playable character.
</item 6>

<item 7>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 7>

<item 8>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 8>

<item 9>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 9>

<item 10>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 10>
=== END ITEM id=204a5caecf9f858c ===

=== ITEM id=0b344ef2a997e06f ===
Gold sentences and facts:
Sentence 0:  The first edition was written by J. Eric Holmes based on Gary Gygax and Dave Arneson's original work.
  Fact 0.0: The first edition was written by J. Eric Holmes.
  Fact 0.1: The first edition was based on the original work of Gary Gygax and Dave Arneson.
Sentence 1: John Eric Holmes M.D. (February 16, 1930 – March 20, 2010) was an associate professor of neurology at the University of Southern California School of Medicine, an author and promoter of fantasy role-playing games, a noted fan and enthusiast of Edgar Rice Burroughs, and an American writer of non-fiction, fantasy and science fiction.
  Fact 1.0: John Eric Holmes M.D. was born on February 16, 1930.
  Fact 1.1: John Eric Holmes M.D. died on March 20, 2010.
  Fact 1.2: John Eric Holmes M.D. was an associate professor of neurology at the University of Southern California School of Medicine.
  Fact 1.3: John Eric Holmes M.D. was an author and promoter of fantasy role-playing games.
  Fact 1.4: John Eric Holmes M.D. was a noted fan and enthusiast of Edgar Rice Burroughs.
  Fact 1.5: John Eric Holmes M.D. was an American writer of non-fiction, fantasy and science fiction.

Retrieved items:
<item 1>
Dungeons &amp; Dragons Basic Game
The Dungeons & Dragons Basic Game is an introductory version of "Dungeons & Dragons" ("D&D") role-playing game packaged in the form of a board game.

 The original game was released in 2004 by Wizards of the Coast and was designed by Jonathan Tweet, one of the "D&D" 3rd edition designers.

 A new version of this game was released in September 2006.
</item 1>

<item 2>
The Lost City (Dungeons &amp; Dragons)
The Lost City (B4) is a "Dungeons & Dragons" adventure module by Tom Moldvay.

 It was first published by TSR in 1982 and was designed as a stand-alone adventure for use with the "Dungeons & Dragons Basic Set".

 The working title for the module was "The Lost City of Cynidecia".

 Moldvay designed the module to give novice Dungeon Masters experience fleshing out adventures and is only partially complete.

 The module is described as a low-level scenario, in which the only hope of the player characters' survival can be found in a ruined city slowly rising out of
the sands.

 The adventure is set inside a huge step pyramid, with the lower pyramid only sketched out and the city itself described with a list of the major areas and a
map.

 The adventure’s main villain is Zargon, a giant one-eyed monster and his minions.

 The entire double pyramid, not including the city, contains over 100 rooms.
</item 2>

<item 3>
Dungeons &amp; Dragons Companion Set
The Dungeons & Dragons Companion Set is an expansion boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 It was first published in 1984 as an expansion to the "Dungeons & Dragons Basic Set".
</item 3>

<item 4>
Dungeons &amp; Dragons Immortals Rules
Dungeons & Dragons Immortals Rules, written by Frank Mentzer, is a boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game first published by
TSR in 1986 as an expansion to the "Basic Set".
</item 4>

<item 5>
Dungeons &amp; Dragons Game (1991 boxed set)
The New Easy-to-Master Dungeons & Dragons Game is an introductory set for the "Dungeons & Dragons" fantasy role-playing game, published by TSR, Inc. in 1991.

 It was a replacement for the previous "Dungeons & Dragons Basic Set", serving to introduce new players to the game, using the rule set previously established.
</item 5>

<item 6>
The Keep on the Borderlands
The Keep on the Borderlands is a "Dungeons & Dragons" module by Gary Gygax, first printed in December 1979.

 In it, player characters are based at a keep and investigate a nearby series of caves that are filled with a variety of monsters.

 It was designed to be used with the "Dungeons & Dragons Basic Set", and was included in the 1979–1982 editions of the "Basic Set".

 It was designed for people new to "Dungeons & Dragons".
</item 6>

<item 7>
Dungeons &amp; Dragons Master Rules
Dungeons & Dragons Master Rules is an expansion boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 It was first published in 1985 as an expansion to the "Basic Set".
</item 7>

<item 8>
GURPS Basic Set
GURPS Basic Set is a hard-bound two volume set written by Steve Jackson, Sean M. Punch, and David L. Pulver.

 The first edition "GURPS Basic Set" was published in 1986.
</item 8>

<item 9>
Dungeons &amp; Dragons Basic Set
The Dungeons & Dragons Basic Set is a set of rulebooks for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 First published in 1977, it saw a handful of revisions and reprintings.

 The first edition was written by J. Eric Holmes based on Gary Gygax and Dave Arneson's original work.

 Later editions were edited by Tom Moldvay, Dave Cook, and Frank Mentzer.
</item 9>

<item 10>
Castle Ravenloft Board Game
Castle Ravenloft Board Game is a 2010 board game published by Wizards of the Coast.

 It was the first game released in the Dungeons & Dragons Adventure System board game series.
</item 10>
=== END ITEM id=0b344ef2a997e06f ===

=== ITEM id=cb9b44cf1817247d ===
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
In 2002 Vassar Brothers Hospital officially became Vassar Brothers Medical Center.

 In September 2016, VBMC announced a new $500 million expansion which would almost double the number of available rooms.

 Construction on the expansion is expected to be completed in 2019.
</item 1>

<item 2>
History of psychiatric hospitals in Pittsburgh
Pittsburgh, Pennsylvania, United States, is home to a plethora of hospitals, be it for the physically ill or, what was defined as the insane and feeble minded.

 In 1891, the St. Margaret Memorial Hospital opened under John H. Shoenberger.

 In 1908, the Pittsburgh Hospital was run under C. A. Fagan, and on June 1, 1889, the South Side Hospital opened under Anna M. Rindlaub.

 Pittsburgh, commonly known as the "Steel City," also built up a reputation for its abundance of hospitals that had opened up within a fairly close time period
of each other.

 During this time, many new mental hospitals were established due to overcrowding.

 In 1899, a building to house the mentally insane, Mayview State Hospital, was constructed; but it was not long before the complex had a reputation “as a place
of sorrow” due to the lack of proper health care patients received during their stay.
</item 2>

<item 3>
Westminster St Margaret and St John
St Margaret was an ancient parish in the City and Liberty of Westminster and the county of Middlesex.

 It included the core of modern Westminster, including the Palace of Westminster and the area around, but not including Westminster Abbey.

 In 1727 it was divided into St Margaret's and St John's, to coincide with the building of the Church of St John the Evangelist, constructed by the Commission
for Building Fifty New Churches in Smith Square to meet the demands of the growing population, but there continued to be a single vestry for the parishes of St
Margaret and St John.

 This was reformed in 1855 by the Metropolis Management Act, and the two parishes formed the Westminster District until 1887.

 In 1889 St Margaret and St John became part of the County of London.

 The vestry was abolished in 1900, to be replaced by Westminster City Council, but St Margaret and St John continued to have a nominal existence until 1922.
</item 3>

<item 4>
St. Elizabeths Hospital
St. Elizabeths Hospital opened in 1855 as the first federally operated psychiatric hospital in the United States.

 Housing over 8,000 patients at its peak in the 1950s, the hospital at one point had a fully functioning medical-surgical unit, a school of nursing, and
accredited internships and psychiatric residencies.

 Its campus was designated a National Historic Landmark in 1990.
</item 4>

<item 5>
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
</item 5>

<item 6>
Montefiore New Rochelle Hospital
Montefiore New Rochelle Hospital (formerly Sound Shore Medical Center) is a community-based, teaching hospital located at 16 Guion Place in the city of New
Rochelle, in Westchester County, New York, and affiliated with the Albert Einstein College of Medicine.

 The hospital opened in 1892 as New Rochelle Hospital.

 On November 6, 2013, Sound Shore was acquired by the Bronx-based Montefiore Medical Center, the University Hospital for the Albert Einstein College of
Medicine, was renamed Montefiore New Rochelle Hospital, and became part of the Montefiore Health System.
</item 6>

<item 7>
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
</item 7>

<item 8>
Providence Hospital (Washington, D.C.)
Providence Hospital is a 408 bed hospital located in the District of Columbia.

 Founded in 1861, it is the longest continuously operating hospital in the District.

 Providence Hospital is a member of Ascension Health, the largest non-profit health care organization in the United States.
</item 8>

<item 9>
Winfred Overholser
Winfred Overholser (1892 – October 6, 1964) was an American psychiatrist, president of the American Psychiatric Association, and for 25 years the superintendent
of St. Elizabeths Hospital, a federal institution for the mentally ill in Washington, D.C.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=cb9b44cf1817247d ===

=== ITEM id=4a8ad719396c8ff4 ===
Gold sentences and facts:
Sentence 0: Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.
  Fact 0.0: Sleepless in Seattle is a 1993 film.
  Fact 0.1: Sleepless in Seattle is an American film.
  Fact 0.2: Sleepless in Seattle is a romantic comedy-drama film.
  Fact 0.3: Sleepless in Seattle was directed by Nora Ephron.
  Fact 0.4: Sleepless in Seattle was co-written by Nora Ephron.
  Fact 0.5: Sleepless in Seattle is based on a story by Jeff Arch.
Sentence 1:  It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber, and Rita Wilson.
  Fact 1.0: Sleepless in Seattle stars Tom Hanks.
  Fact 1.1: Sleepless in Seattle stars Meg Ryan.
  Fact 1.2: Sleepless in Seattle has a supporting cast featuring Bill Pullman.
  Fact 1.3: Sleepless in Seattle has a supporting cast featuring Ross Malinger.
  Fact 1.4: Sleepless in Seattle has a supporting cast featuring Rob Reiner.
  Fact 1.5: Sleepless in Seattle has a supporting cast featuring Rosie O'Donnell.
  Fact 1.6: Sleepless in Seattle has a supporting cast featuring Gaby Hoffmann.
  Fact 1.7: Sleepless in Seattle has a supporting cast featuring Victor Garber.
  Fact 1.8: Sleepless in Seattle has a supporting cast featuring Rita Wilson.
Sentence 2: Roseann O'Donnell (born March 21, 1962) is an American comedian, actress, author, and television personality.
  Fact 2.0: Roseann O'Donnell was born on March 21, 1962.
  Fact 2.1: Roseann O'Donnell is an American comedian.
  Fact 2.2: Roseann O'Donnell is an American actress.
  Fact 2.3: Roseann O'Donnell is an American author.
  Fact 2.4: Roseann O'Donnell is an American television personality.

Retrieved items:
<item 1>
The 'Burbs
The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy
Schaal and Henry Gibson.

 The film was written by Dana Olsen, who also has a cameo in the movie.

 The film pokes fun at suburban environments and their eccentric dwellers.
</item 1>

<item 2>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 2>

<item 3>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 3>

<item 4>
In My Sleep
In My Sleep is a 2010 suspense thriller film written, directed and produced by film director Allen Wolf.

 It stars Philip Winchester, Lacey Chabert, Tim Draxl, Abigail Spencer and Kelly Overton.

 It also features Kirsten Vangsness from Criminal Minds in a cameo role and Tony Hale from Veep in a cameo as well.

 The story is about a massage therapist with chronic insomnia who fears he may have murdered a good friend while sleepwalking.
</item 4>

<item 5>
Carrie Fisher
Carrie Frances Fisher (October 21, 1956 – December 27, 2016) was an American actress, writer, and humorist.

 Fisher was known for playing Princess Leia in the "Star Wars" films; a role for which she was nominated for three Saturn Awards.

 Her other film roles included "Shampoo" (1975), "The Blues Brothers" (1980), "Hannah and Her Sisters" (1986), "The 'Burbs" (1989), "When Harry Met Sally..."
(1989), "Soapdish" (1991) and "The Women" (2008 film).

 She was nominated twice for the Primetime Emmy Award for Outstanding Guest Actress in a Comedy Series for her performances on the television series "30 Rock"
and "Catastrophe".

 She was posthumously made a Disney Legend in 2017.
</item 5>

<item 6>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 6>

<item 7>
Barry Sandler
Barry Sandler (born February 23, 1947 in Buffalo, New York) is an American screenwriter and film producer.

 His career has spanned several decades, with the 1980s being his most prolific.

 The openly gay Sandler is perhaps best known for writing the 1982 film "Making Love", the first mainstream Hollywood film to deal seriously with issues of
homosexuality and coming out.

 Sandler discussed "Making Love" in the 1995 documentary film "The Celluloid Closet".
</item 7>

<item 8>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 8>

<item 9>
Matthew Perry
Matthew Langford Perry (born August 19, 1969) is a Canadian-American actor, best known for his role as Chandler Bing on the long-running NBC television sitcom
"Friends", as well as his portrayal of Ron Clark in the 2006 television movie "The Ron Clark Story".
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
=== END ITEM id=4a8ad719396c8ff4 ===

=== ITEM id=c2da9c95f9e62b35 ===
Gold sentences and facts:
Sentence 0: The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic Beach.
  Fact 0.0: The Nassau Herald is a weekly newspaper.
  Fact 0.1: The Nassau Herald serves the Five Towns communities of Nassau County.
  Fact 0.2: The Five Towns communities of Nassau County are Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood, and Atlantic Beach.
Sentence 1: Lawrence is a village in Nassau County, New York in the United States.
  Fact 1.0: Lawrence is a village.
  Fact 1.1: Lawrence is in Nassau County, New York, United States.
Sentence 2:  As of the 2010 United States Census, the village population was 6,483.
  Fact 2.0: As of the 2010 United States Census, the population of the village of Lawrence was 6,483.

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
Great Neck School District
The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.

 It also serves students in some parts of Manhasset and New Hyde Park.

 It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
</item 2>

<item 3>
Great Neck, New York
Great Neck is a region on Long Island, New York, that covers a peninsula on the North Shore of Long Island, which includes the villages of Great Neck, Great
Neck Estates, Great Neck Plaza, and others, as well as an area south of the peninsula near Lake Success and the border territory of Queens.

 The incorporated village of Great Neck had a population of 9,989 at the 2010 census, while the larger Great Neck area comprises a residential community of some
40,000 people in nine villages and hamlets in the town of North Hempstead, of which Great Neck is the northwestern quadrant.

 Great Neck has five ZIP Codes (11020–11024), and one school district.
</item 3>

<item 4>
Lawrence, Nassau County, New York
Lawrence is a village in Nassau County, New York in the United States.

 As of the 2010 United States Census, the village population was 6,483.
</item 4>

<item 5>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 5>

<item 6>
The Hudson Reporter
The Hudson Reporter is a newspaper chain based in Hudson County, New Jersey.

 It is the only weekly newspaper chain in Hudson County and one of only two newspaper companies in this busy metropolitan area.

 "The Hudson Reporter" publications focus on local politics and community news.

 In addition to articles written by the staff, the papers print readers' letters to the editor.

 The oldest newspaper in the chain is the "Hoboken Reporter", founded in 1983.
</item 6>

<item 7>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 7>

<item 8>
Anton Media Group
Anton Media Group, formerly Anton Community Newspapers, and also known by its legal name Long Island Community Newspapers, Inc. is a print media company based
in Mineola, NY on Long Island in Nassau County and produces 17 weekly newspapers in Nassau County as well as "Long Island Weekly" and special sections such as
Healthy Living, Camp & School, and Dining Guides.

 Anton Community Newspapers is one of the largest privately owned newspaper companies in New York State.
</item 8>

<item 9>
Nassau Herald
The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic
Beach.

 It is part of the Long Island Herald newspaper chain, which includes The Jewish Star and The Oyster Bay Guardian is owned by Richner Communications, and covers
Nassau County, New York.
</item 9>

<item 10>
Floral Park, New York
Floral Park is an incorporated village in Nassau County, New York, United States, on Long Island.

 The neighborhood of Floral Park in the New York City borough of Queens, is adjacent to the village.

 The village is at the western border of Nassau County, and is located mainly in the Town of Hempstead, while the section north of Jericho Turnpike is within
the Town of North Hempstead.

 The population as of the US Census of 2010 is 15,863.
</item 10>
=== END ITEM id=c2da9c95f9e62b35 ===
