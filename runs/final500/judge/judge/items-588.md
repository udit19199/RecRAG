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

=== ITEM id=ff4ad07848c4134a ===
Gold sentences and facts:
Sentence 0: Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.
  Fact 0.0: Margaret Landon was born September 7, 1903.
  Fact 0.1: Margaret Landon died December 4, 1993.
  Fact 0.2: Margaret Landon was an American writer.
  Fact 0.3: Margaret Landon is best remembered for 'Anna and the King of Siam'.
  Fact 0.4: 'Anna and the King of Siam' is Margaret Landon's best-selling novel, published in 1944.
  Fact 0.5: 'Anna and the King of Siam' is a novel about the life of Anna Leonowens.
  Fact 0.6: 'Anna and the King of Siam' eventually sold over a million copies.
  Fact 0.7: 'Anna and the King of Siam' was translated into more than twenty languages.
Sentence 1:  In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.
  Fact 1.0: In 1950, Margaret Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II.
  Fact 1.1: Richard Rodgers and Oscar Hammerstein II created the musical 'The King and I' from Margaret Landon's book.
Sentence 2:  A later work, "Never Dies the Dream", appeared in 1949.
  Fact 2.0: 'Never Dies the Dream' is a later work.
  Fact 2.1: 'Never Dies the Dream' appeared in 1949.
Sentence 3: Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.
  Fact 3.0: Dame Elizabeth Rosemond Taylor, DBE, was born February 27, 1932.
  Fact 3.1: Dame Elizabeth Rosemond Taylor, DBE, died March 23, 2011.
  Fact 3.2: Dame Elizabeth Rosemond Taylor was a British-American actress.
  Fact 3.3: Dame Elizabeth Rosemond Taylor was a businesswoman.
  Fact 3.4: Dame Elizabeth Rosemond Taylor was a humanitarian.
Sentence 4:  She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.
  Fact 4.0: Elizabeth Taylor began her career as a child actress in the early 1940s.
  Fact 4.1: Elizabeth Taylor was one of the most popular stars of classical Hollywood cinema in the 1950s.
Sentence 5:  She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.
  Fact 5.0: Elizabeth Taylor continued her career successfully into the 1960s.
  Fact 5.1: Elizabeth Taylor remained a well known public figure for the rest of her life.
Sentence 6:  In 1999, the American Film Institute named her the seventh-greatest female screen legend.
  Fact 6.0: In 1999, the American Film Institute named Elizabeth Taylor the seventh-greatest female screen legend.

Retrieved items:
<item 1>
Anna and the King of Siam (novel)
Anna and the King is a 1944 semi-fictionalized biographical novel by Margaret Landon.
</item 1>

<item 2>
Elizabeth Taylor
Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.

 She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.

 She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.

 In 1999, the American Film Institute named her the seventh-greatest female screen legend.
</item 2>

<item 3>
Anna and the King of Siam (film)
Anna and the King of Siam is a 1946 drama film directed by John Cromwell.

 An adaptation of the 1944 novel of the same name by Margaret Landon, it was based on the fictionalized diaries of Anna Leonowens, an Anglo-Indian woman who
claimed to be British and became governess in the Royal Court of Siam (now modern Thailand) during the 1860s.

 Darryl F. Zanuck read Landon's book in galleys and immediately bought the film rights.
</item 3>

<item 4>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 4>

<item 5>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 5>

<item 6>
At Land
At Land (1944) is a 15-minute silent experimental film written, directed by, and starring Maya Deren.

 It has a dream-like narrative in which a woman, played by Deren, is washed up on a beach and goes on a strange journey encountering other people and other
versions of herself.

 Deren once said that the film is about the struggle to maintain one's personal identity.
</item 6>

<item 7>
Margaret Landon
Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of
the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.

 In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.

 A later work, "Never Dies the Dream", appeared in 1949.
</item 7>

<item 8>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 8>

<item 9>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 9>

<item 10>
Sam Taylor (director)
Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.

 Taylor is best known for his comedic directorial work with Harold Lloyd and Mary Pickford, and also later worked with Laurel and Hardy.
</item 10>
=== END ITEM id=ff4ad07848c4134a ===

=== ITEM id=728187949f8ec9d4 ===
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
Duke of Bedford
Duke of Bedford (named after Bedford, England) is a title that has been created six times (for five distinct people) in the Peerage of England.

 The first and second creations came in 1414 in favour of Henry IV's third son, John, who later served as regent of France.

 He was made Earl of Kendal at the same time and was made Earl of Richmond later the same year.

 The titles became extinct on his death in 1435.

 The third creation came in 1470 in favour of George Neville, nephew of Warwick the Kingmaker.

 He was deprived of the title by Act of Parliament in 1478.

 The fourth creation came 1478 in favour of George, the third son of Edward IV.

 He died the following year at the age of two.

 The fifth creation came in 1485 in favour of Jasper Tudor, half-brother of Henry VI and uncle of Henry VII.

 He had already been created Earl of Pembroke in 1452.

 However, as he was a Lancastrian, his title was forfeited between 1461 and 1485 during the predominance of the House of York.
</item 1>

<item 2>
Canada under British rule
Although confederation in 1867 led to an enlarged Dominion with increased autonomy over domestic affairs, Canada still remained a colony within the British
Empire and was thus subordinate to the British Parliament until the enactment of the Statute of Westminster in 1931.

 This statute recognized Canada as an independent peer coequal with the United Kingdom, and thus provided the Parliament of Canada with legislative sovereignty
over all federal matters except the power to change the constitutional laws of Canada which remained under the purview of the Parliament of the United Kingdom.

 Canada's final vestige of legal dependence on the United Kingdom was terminated in 1982 with the enactment of the Canada Act, subsequently providing Canada
with full legal sovereignty completely independent of the United Kingdom.
</item 2>

<item 3>
House of Saxe-Coburg and Gotha
The House of Saxe-Coburg and Gotha ( ; German: "") is a German dynasty that ruled the duchy of Saxe-Coburg and Gotha, which was one of the Ernestine duchies.

 It is a cadet branch of the Saxon House of Wettin.
</item 3>

<item 4>
Duke of Bedford
He regained the earldom in 1485 when his nephew Henry VII came to the throne and was elevated to the dukedom the same year.

 He had no legitimate children and the titles became extinct on his death in 1495.
</item 4>

<item 5>
Principality of Calenberg
The Principality of Calenberg was a dynastic division of the Welf duchy of Brunswick-Lüneburg established in 1432.

 Calenberg was ruled by the House of Hanover from 1635 onwards; the princes received the ninth electoral dignity of the Holy Roman Empire in 1692.

 Their territory became the nucleus of the Electorate of Hanover, ruled in personal union with the Kingdom of Great Britain from 1714 onwards.

 The principality received its name from Calenberg Castle, a residence of the Brunswick dukes.
</item 5>

<item 6>
House of Hanover
The House of Hanover (or the Hanoverians ; German: "Haus Hannover" ) is a German royal dynasty that ruled the Electorate and then the Kingdom of Hanover, and
that also provided monarchs of Great Britain and Ireland from 1714 and ruled the United Kingdom until the death of Queen Victoria in 1901.

 Upon Victoria's death, the British throne passed to her eldest son Edward VII, a member of the House of Saxe-Coburg and Gotha through his father.
</item 6>

<item 7>
Succession to the British throne
Succession to the British throne is determined by descent, gender (for people born before October 2011), legitimacy, and religion.

 Under common law, the Crown is inherited by a sovereign's children or by a childless sovereign's nearest collateral line.

 The Bill of Rights 1689 and the Act of Settlement 1701, restrict succession to the throne to the legitimate Protestant descendants of Sophia of Hanover that
are in "communion with the Church of England".

 Spouses of Roman Catholics were disqualified from 1689 until the law was amended in 2015.

 Protestant descendants of those excluded for being Roman Catholics are eligible.
</item 7>

<item 8>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 8>

<item 9>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 9>

<item 10>
House of Oldenburg
The House of Oldenburg is a European dynasty of North German origin.

 It is one of Europe's most influential royal houses, with branches that rule or have ruled in Denmark, Iceland, Greece, Norway, Russia, Sweden, Schleswig,
Holstein, and Oldenburg.

 The current Queen of Denmark and King of Norway, the former King of Greece, the consorts of Greece and the United Kingdom, as well as the first twelve names in
the line of succession to the British throne, are all patrilineal members of the Glücksburg branch of this house.
</item 10>
=== END ITEM id=728187949f8ec9d4 ===

=== ITEM id=a63dbc1ecb2d79ea ===
Gold sentences and facts:
Sentence 0: Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space as part of the Soviet Union's Interkosmos programme.
  Fact 0.0: Sigmund Werner Paul Jähn was born on 13 February 1937.
  Fact 0.1: Sigmund Werner Paul Jähn is a German cosmonaut.
  Fact 0.2: Sigmund Werner Paul Jähn is a pilot.
  Fact 0.3: In 1978 Sigmund Werner Paul Jähn became the first East German to fly in space.
  Fact 0.4: In 1978 Sigmund Werner Paul Jähn became the first German native to fly in space.
  Fact 0.5: The 1978 flight by Sigmund Werner Paul Jähn was part of the Soviet Union's Interkosmos programme.
Sentence 1: Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.
  Fact 1.0: Arne Christer Fuglesang was born on March 18, 1957.
  Fact 1.1: Arne Christer Fuglesang was born in Stockholm.
  Fact 1.2: Arne Christer Fuglesang is a Swedish physicist.
  Fact 1.3: Arne Christer Fuglesang is an ESA astronaut.
Sentence 2:  He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
  Fact 2.0: Arne Christer Fuglesang was first launched aboard the STS-116 Space Shuttle mission.
  Fact 2.1: That launch was on December 10, 2006, at 01:47 GMT.
  Fact 2.2: Arne Christer Fuglesang became the first Swedish citizen in space.

Retrieved items:
<item 1>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 1>

<item 2>
Samantha Cristoforetti
Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.

 She holds the record for the longest uninterrupted spaceflight of a European astronaut (199 days, 16 hours), and until June 2017 held the record for the
longest single space flight by a woman until this was broken by Peggy Whitson.

 She is also the first Italian woman in space.

 Samantha Cristoforetti is also known as the first person who brewed an espresso coffee in space.
</item 2>

<item 3>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 3>

<item 4>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 4>

<item 5>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 5>

<item 6>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 6>

<item 7>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 7>

<item 8>
Sigmund Jähn
Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space
as part of the Soviet Union's Interkosmos programme.
</item 8>

<item 9>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 9>

<item 10>
Eberhard Köllner
Eberhard Köllner (born 29 September 1939 in Stassfurt, Germany) was selected for Soyuz 31 as the backup for Sigmund Jähn.
</item 10>
=== END ITEM id=a63dbc1ecb2d79ea ===

=== ITEM id=77517805a7c300af ===
Gold sentences and facts:
Sentence 0: Georg Ferdinand Ludwig Philipp Cantor ( ; ] ; March 3 [O.S. February 19] 1845  – January 6, 1918) was a German mathematician.
  Fact 0.0: Georg Ferdinand Ludwig Philipp Cantor was born on March 3, 1845.
  Fact 0.1: The birth date of Georg Ferdinand Ludwig Philipp Cantor is also given as February 19, 1845 in the Old Style calendar.
  Fact 0.2: Georg Ferdinand Ludwig Philipp Cantor died on January 6, 1918.
  Fact 0.3: Georg Ferdinand Ludwig Philipp Cantor was a German mathematician.
Sentence 1: Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian mathematician.
  Fact 1.0: Vladimir Igorevich Arnold is also spelled Arnol'd.
  Fact 1.1: Vladimir Igorevich Arnold's name in Russian is Влади́мир И́горевич Арно́льд.
  Fact 1.2: Vladimir Igorevich Arnold was born on 12 June 1937.
  Fact 1.3: Vladimir Igorevich Arnold died on 3 June 2010.
  Fact 1.4: Vladimir Igorevich Arnold was a Soviet mathematician.
  Fact 1.5: Vladimir Igorevich Arnold was a Russian mathematician.

Retrieved items:
<item 1>
Benedict Arnold
Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American
Continental Army, and later defected to the British Army.

 While a general on the American side, he obtained command of the fortifications at West Point, New York (which after 1802 would become the site of the U.S.
Military Academy), overlooking the cliffs at the Hudson River (upriver from British-occupied New York City), and planned to surrender them to British forces.

 This plan was exposed in September 1780.

 He was commissioned into the British Army as a brigadier general.
</item 1>

<item 2>
Otto Struve
Otto Struve (August 12, 1897 – April 6, 1963) was a astronomer.

 In Russian, his name is sometimes given as Otto Lyudvigovich Struve (Отто Людвигович Струве); however, he spent most of his life and his entire scientific
career in the United States.

 Otto was the descendant of famous astronomers of the Struve family; he was the son of Ludwig Struve, grandson of Otto Wilhelm von Struve and great-grandson of
Friedrich Georg Wilhelm von Struve.

 He was also the nephew of Karl Hermann Struve.
</item 2>

<item 3>
Cantor–Bernstein theorem
In set theory and order theory, the Cantor–Bernstein theorem states that the cardinality of the second type class, the class of countable order types, equals
the cardinality of the continuum.

 It was used by Felix Hausdorff and named by him after Georg Cantor and Felix Bernstein.

 Cantor constructed a family of countable order types with the cardinality of the continuum, and in his 1901 inaugural dissertation Bernstein proved that such a
family can have no higher cardinality.
</item 3>

<item 4>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 4>

<item 5>
Georg Cantor's first set theory article
Georg Cantor's first set theory article was published in 1874 and contains the first theorems of transfinite set theory, which studies infinite sets and their
properties.

 One of these theorems is "Cantor's revolutionary discovery" that the set of all real numbers is uncountably, rather than countably, infinite.

 This theorem is proved using Cantor's first uncountability proof, which differs from the more familiar proof using his diagonal argument.

 The title of the article, "On a Property of the Collection of All Real Algebraic Numbers" ("Ueber eine Eigenschaft des Inbegriffes aller reellen algebraischen
Zahlen "), refers to its first theorem: the set of real algebraic numbers is countable.
</item 5>

<item 6>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 6>

<item 7>
Georg Cantor
Georg Ferdinand Ludwig Philipp Cantor ( ; ] ; March 3 [O.S. February 19] 1845  – January 6, 1918) was a German mathematician.

 He invented set theory, which has become a fundamental theory in mathematics.

 Cantor established the importance of one-to-one correspondence between the members of two sets, defined infinite and well-ordered sets, and proved that the
real numbers are more numerous than the natural numbers.

 In fact, Cantor's method of proof of this theorem implies the existence of an "infinity of infinities".

 He defined the cardinal and ordinal numbers and their arithmetic.

 Cantor's work is of great philosophical interest, a fact of which he was well aware.
</item 7>

<item 8>
Alfred von Arneth
Alfred Ritter von Arneth (10 July 181930 July 1897) was an Austrian historian.

 His principal scholarly work is a ten-volume biography of the Habsburg Empress Maria Theresa, first published in installments from 1863 to 1879 and still
regarded as the standard work on the subject.

 Born at Vienna, he was the son of Joseph Calasanza von Arneth (1791–1863), a well-known historian and archaeologist, who wrote a history of the Austrian Empire
(Vienna, 1827) and several works on numismatics and brother of Doctor Franz Hektor von Arneth (1818–1907).
</item 8>

<item 9>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 9>

<item 10>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 10>
=== END ITEM id=77517805a7c300af ===

=== ITEM id=a68960d7fff7aef5 ===
Gold sentences and facts:
Sentence 0: The University of Delaware (colloquially "UD") is the largest university in Delaware.
  Fact 0.0: The University of Delaware is colloquially called 'UD'.
  Fact 0.1: The University of Delaware is the largest university in Delaware.
Sentence 1: The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.
  Fact 1.0: The University of Debrecen's Hungarian name is 'Debreceni Egyetem'.
  Fact 1.1: The University of Debrecen is a university.
  Fact 1.2: The University of Debrecen is located in Debrecen, Hungary.

Retrieved items:
<item 1>
Ellendale, Delaware
Ellendale is a town in Sussex County, Delaware, United States.

 The population was 381 at the 2010 census, an increase of 16.5% since 2000.

 It is part of the Salisbury, Maryland-Delaware Metropolitan Statistical Area.

 Ellendale is the "Gateway to Delaware's Resort Beaches" because it is the town located on U.S. Highway 113, the resort area's westernmost border, and Delaware
Route 16, the resort area's northernmost border with the eastern border being the Delaware Bay and Atlantic Ocean and the southern border being the state line
with Maryland.

 Ellendale is home to the Philadelphia Bible College and the Harbor Christian Academy.
</item 1>

<item 2>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 2>

<item 3>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 3>

<item 4>
Delaware–William &amp; Mary football rivalry
The Delaware–William & Mary football rivalry between the Delaware Fightin' Blue Hens and the William & Mary Tribe is a match-up between two public universities,
the University of Delaware and the College of William and Mary, that are also members of the Colonial Athletic Association.

 Both schools have academic reputations that have labeled them as Public Ivies.

 Both schools are also Colonial Colleges having been founded before the United States became independent in 1776; the College of William and Mary was founded in
1693 and the University of Delaware's predecessor school was founded in 1743.
</item 4>

<item 5>
Graph query row:
university: University of Delaware
located_in: Delaware
country: United States
country_relationship: LOCATED_IN
</item 5>

<item 6>
Rutgers University–Camden
Rutgers University–Camden is one of three regional campuses of Rutgers University, New Jersey's public research university.

 It is located in Camden, New Jersey, United States.

 Founded in the 1920s, Rutgers–Camden began as an amalgam of the South Jersey Law School and the College of South Jersey.

 It is the southernmost of the three regional campuses of Rutgers—the others being located in New Brunswick and Newark.

 The city of Camden is located on the Delaware River, east of Philadelphia.
</item 6>

<item 7>
University of Debrecen
The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.

 It is the oldest continuously operating institution of higher education in Hungary (since 1538).

 The university has a well established programme in the English language for international students, particularly in the Medical field, which first established
education in English in 1986.

 There are nearly 4000 international students studying at the university.
</item 7>

<item 8>
University of Delaware Botanic Gardens
The University of Delaware Botanic Gardens are botanical gardens and an arboretum located on the campus of the University of Delaware, in Newark, Delaware,
United States.

 The gardens are open to the public without charge.
</item 8>

<item 9>
College Park, Delaware
College Park is an unincorporated community in New Castle County, Delaware, United States.

 College Park is located along Delaware Route 896 southwest of Newark.
</item 9>

<item 10>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 10>
=== END ITEM id=a68960d7fff7aef5 ===

=== ITEM id=9bb00bc44e2d0166 ===
Gold sentences and facts:
Sentence 0:  It consists of three shots, two Sourz Apple and one Cherry Brandy.
  Fact 0.0: The described drink consists of three shots.
  Fact 0.1: The described drink has two shots of Sourz Apple.
  Fact 0.2: The described drink has one shot of Cherry Brandy.
Sentence 1:  Each shot is taken from an average shot glass which is around 1.5 ounces (roughly 45 ml).
  Fact 1.0: Each shot is taken from an average shot glass.
  Fact 1.1: An average shot glass holds around 1.5 ounces (roughly 45 ml).
Sentence 2: The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.
  Fact 2.0: The flaming volcano is a large tropical group cocktail.
  Fact 2.1: The flaming volcano is made of rum, brandy, pineapple juice, orange juice and almond syrup.
  Fact 2.2: The flaming volcano is sometimes made with other ingredients.
Sentence 3:  It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about 32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.
  Fact 3.0: The flaming volcano is usually a multi-user drink.
  Fact 3.1: The flaming volcano is served to a group in a special ritual vessel known as a volcano bowl.
  Fact 3.2: A volcano bowl is a decorative ceramic bowl.
  Fact 3.3: A volcano bowl typically has a capacity of about 32 USoz.
  Fact 3.4: A volcano bowl is designed with a rising central hub feature.
  Fact 3.5: The rising central hub feature of a volcano bowl usually resembles a crude volcano cone.

Retrieved items:
<item 1>
Volcano bowl
Volcano bowls are ceramic drinkware originally associated with mid-20th century American tiki bars and tropical-themed restaurants.

 Drinks served in volcano bowls are typically rum-based, mixed with tropical fruit juices and other liquors such as brandy, vodka, and triple sec, and garnished
with fruit.

 The Flaming Volcano cocktail is especially associated with this drinkware.
</item 1>

<item 2>
Hotlum Glacier
The Hotlum Glacier is a glacier situated on the northeast flank of Mount Shasta, in the US state of California.

 It is the largest and most voluminous glacier in California, although not as thick or long as the nearby Whitney Glacier.

 The Hotlum Glacier flows from a large cirque on the northeast side of Mount Shasta's main summit below the Hotlum Headwall at roughly 13600 ft .

 It flows northeastward down the steep slope, forming three lobes which terminate near 10400 ft .
</item 2>

<item 3>
Flaming Moe's
"Flaming Moe's" is the tenth episode of "The Simpsons"' third season.

 It first aired on the Fox network in the United States on November 21, 1991.

 In the episode, Homer tells Moe Szyslak of a secret alcoholic cocktail that includes cough medicine and fire that he calls "Flaming Homer".

 Moe steals the recipe from Homer, renames the drink the "Flaming Moe" and begins selling it at his tavern.

 The drink is a success and boosts business and patronage, but Homer is angry at Moe for his betrayal, and seeks revenge.
</item 3>

<item 4>
Volcanic bomb
A volcanic bomb is a mass of molten rock (tephra) larger than 64 mm (2.5 inches) in diameter, formed when a volcano ejects viscous fragments of lava during an
eruption.

 They cool into solid fragments before they reach the ground.

 Because volcanic bombs cool after they leave the volcano, they are extrusive igneous rocks.

 Volcanic bombs can be thrown many kilometres from an erupting vent, and often acquire aerodynamic shapes during their flight.

 Bombs can be extremely large; the 1935 eruption of Mount Asama in Japan expelled bombs measuring 5–6 m in diameter up to 600 m from the vent.

 Volcanic bombs are a significant volcanic hazard, and can cause severe injuries and death to people in an eruption zone.

 One such incident occurred at Galeras volcano in Colombia in 1993; six people near the summit were killed and several seriously injured by lava bombs when the
volcano erupted unexpectedly.
</item 4>

<item 5>
Tweed Volcano
Tweed Volcano is a partially eroded Early Miocene shield volcano located in northeastern New South Wales, which formed when this region of Australia passed over
the East Australia hotspot around 23 million years ago.

 Mount Warning, Lamington Plateau and the Border Ranges between New South Wales and Queensland are among the remnants of this volcano that was originally over
100 km in diameter and nearly twice the height of Mount Warning today, at 1156 m .

 Despite its size, Tweed Volcano was not a supervolcano; other shield volcanoes - such as on Hawaii - are much larger.

 In the 23 million years since the volcano was active, erosion has been extensive, forming a large erosion caldera around the volcanic plug of Mount Warning.

 Its erosion caldera is the largest in the Southern Hemisphere.
</item 5>

<item 6>
Flaming volcano
The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.

 It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about
32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.

 The cone includes a second, much smaller central pit or "crater" which is partially filled with rum or other flammable high-alcohol liquor.

 The "crater" liquor is then ignited, creating a mild volcanic ambience with its central blue flame.
</item 6>

<item 7>
Apple-Kneel
The Apple-Kneel is a cocktail.

 It consists of three shots, two Sourz Apple and one Cherry Brandy.

 Each shot is taken from an average shot glass which is around 1.5 ounces (roughly 45 ml).

 Firstly, you put two shots of Sourz Apple into a glass and then put in one shot of Cherry Brandy in and mix.

 It is customary to sip this cocktail.

 It can be served 'neat' or 'on the rocks'.
</item 7>

<item 8>
Flaming Doctor Pepper
A Flaming Doctor Pepper is a flaming cocktail said to taste like the soft drink Dr Pepper, although Dr Pepper is not one of its ingredients.
</item 8>

<item 9>
Man-eating tree
Man-eating tree can refer to any of the various legendary carnivorous plants large enough to kill and consume a person or other large animal.

 The carnivorous plant with the largest known traps is probably Nepenthes rajah, which produces pitchers up to 41 cm tall with a volume up to 3.5 l .

 The pitcher of this species are designed to trap arthropods.

 However, the same bait may also attract rodents like the summit rat ("Rattus baluensis") and the Mountain treeshrew ("Tupaia montana").

 Only very rarely will the rodents fall into the large pitchers of this species.

 Other large carnivorous plants that have similar properties include Nepenthes robcantleyi and Nepenthes attenboroughi.
</item 9>

<item 10>
Flaming beverage
Flaming beverages include cocktails and other mixed drinks that contain flammable, high-proof alcohol, which is ignited prior to consumption.

 The alcohol may be an integral part of the drink, or it may be floated as a thin layer across the top of the drink.

 The flames are mostly for dramatic flair.

 However, in combination with certain ingredients, the flavor of the drink is altered.

 Some flavors are enhanced, and it may impart a toasted flavor to some drinks.
</item 10>
=== END ITEM id=9bb00bc44e2d0166 ===

=== ITEM id=3f3877016c14cd1a ===
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
Wayne Coles-Janess
Wayne Coles-Janess is an Australian producer, writer and director of drama and documentary film and TV programs.

 Based in Melbourne, Australia, he has produced documentaries about frontier places in the country.

 He has also made some documentaries in several international locations, including during times of war.
</item 1>

<item 2>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 2>

<item 3>
Bougainville – Our Island Our Fight
Bougainville – Our Island Our Fight is a 1998 Australian documentary film.

 It was produced and directed by Wayne Coles-Janess.
</item 3>

<item 4>
Life at the End of the Rainbow
Life at the End of the Rainbow is a 2002 documentary by Australian filmmaker Wayne Coles-Janess about the small farming community of Rainbow, population 500,
which lies on the edge of the Big Desert, North Western Victoria, Australia.

 It is 55 minutes long.
</item 4>

<item 5>
The Boys in Company C
The Boys in Company C, directed by Sidney J. Furie, starring Stan Shaw, Andrew Stevens, Craig Wasson, James Canning, and Michael Lembeck, is a 1978 film about
United States Marine Corps recruits preparing for duty, and their subsequent combat in the Vietnam War.

 It was among the first Vietnam War films to appear after the Vietnam Era, and was also the first role for R. Lee Ermey of "Full Metal Jacket" fame.

 "The Boys in Company C" is the first in Furie's Vietnam War motion picture trilogy, followed by 2001's "Under Heavy Fire" and 2006's "The Veteran".
</item 5>

<item 6>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 6>

<item 7>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 7>

<item 8>
The Coconut Revolution
The Coconut Revolution is a 2001 multi-award winning documentary film about the struggle of the indigenous peoples of Bougainville Island during the
Bougainville Civil War.

 The movement is described as the "world's first successful eco-revolution" and has drawn parallels with the conflict depicted in the 2009 film, "Avatar".
</item 8>

<item 9>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 9>

<item 10>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 10>
=== END ITEM id=3f3877016c14cd1a ===

=== ITEM id=8510ea1504f8654e ===
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
Litbits
Litbits is a literary discussion radio programme and podcast, hosted by academic Adam Smyth and literary journalist James Kidd.

 Guests on the programme have included Polly Russell, Paul Myerscough, Alex von Tunzelmann, Samantha Ellis, Jonathan Beckman, Hallie Rubenhold, Steve Rose, Joe
Brooker, Dennis Duncan, Dinah Roe, Giles Milton and Liane Strauss.

 Subjects covered include Rooms and Procrastination.
</item 1>

<item 2>
Andrew Benjamin
Andrew Benjamin (born 1952, Australia) is an Australian philosopher.

 Currently he holds two academic posts: Distinguished Professor of Philosophy and the Humanities at Kingston University, London and Professor of Philosophy and
Jewish Thought in the Department of Philosophy and the Australian Centre for Jewish Civilization at Monash University, Melbourne, Australia.

 Benjamin first came to critical attention with his writings in continental philosophy, writing articles and editing books on the thinking of Martin Heidegger,
Theodor Adorno, Walter Benjamin, Julia Kristeva and Jean-François Lyotard.

 Benjamin has become involved in the field of architecture, to the extent that he has also taught in various schools of architecture in UK, US and Australia.
</item 2>

<item 3>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 3>

<item 4>
Inklings
The Inklings was an informal literary discussion group associated with the University of Oxford, England, for nearly two decades between the early 1930s and
late 1949.

 The Inklings were literary enthusiasts who praised the value of narrative in fiction and encouraged the writing of fantasy.
</item 4>

<item 5>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 5>

<item 6>
James F. Light
James F. Light was an American literary scholar, university vice president, and provost.

 During his academic career, he helped revive the works of satirist Nathanael West, with the first book length critical study of his work, "Nathanael West: An
Interpretive Study", (Northwestern Univ.

 Press, 1961).

 He was also the leading authority on John William De Forest, the early American realist whose work he critiqued in "John William De Forest" (Twayne Pub.,
1965), and he wrote extensively on J. D. Salinger, Robert Penn Warren and others.
</item 6>

<item 7>
Eagle Street College
The Eagle Street College was an informal literary society established in 1885 at the home of James William Wallace in Eagle Street, Bolton, to read and discuss
literary works, particularly the poetry of Walt Whitman, (1819–91).

 The group subsequently became known as the Bolton Whitman Fellowship or Whitmanites.

 Its founder members were Wallace, Dr John Johnston and Fred Wild.

 The group held an annual 'Whitman Day' celebration around the 31 May which was the poet's birthday.
</item 7>

<item 8>
Falmouth University
Falmouth University (Cornish: Pennskol Aberfala ) is a specialist University for the creative industries based in Falmouth and Penryn, Cornwall, England.

 Founded in 1902, it had previously been the Falmouth School of Art, Falmouth College of Art and Design and then Falmouth College of Arts until it received
taught degree-awarding powers, and the right to use the title "University College", in March 2005.

 In April 2008, University College Falmouth merged with Dartington College of Arts, adding a range of Performance courses to its portfolio.

 On 27 November 2012, a communication was released to the staff and students and local press that 'University College Falmouth is to be granted full university
status in a move that will further its ambition to become one of the top five arts universities in the world.'

 On 9 December 2012, the University College was officially granted full university status by the Privy Council.

 Falmouth University was judged by The Sunday Times to be the UK’s top arts university from 2015-2017.
</item 8>

<item 9>
Alpha Gamma
Alpha Gamma (ΑΓ) was an American collegiate fraternity.

 It was founded at Cumberland University in Lebanon, Tennessee in 1867.

 About 21 chapters developed, with prominent chapters at Washington & Jefferson College, Trinity University, Mercersburg College, Southwestern Presbyterian
University, Cumberland University, and the West Virginia University.

 The fraternity's badge consisted of a golden shield with a globe, six stars, and a pennant with "ΑΓ."
</item 9>

<item 10>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 10>
=== END ITEM id=8510ea1504f8654e ===

=== ITEM id=78577320ce777739 ===
Gold sentences and facts:
Sentence 0:  He appeared in 57 films between 1919 and 1932.
  Fact 0.0: The person described appeared in 57 films.
  Fact 0.1: The person described appeared in 57 films between 1919 and 1932.
Sentence 1:  He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
  Fact 1.0: The person described directed 54 films.
  Fact 1.1: The person described directed these films between 1920 and 1933.
  Fact 1.2: The person's directed films include one Harry Langdon short.
  Fact 1.3: The person's directed films include two of the Tay Garnett-penned comedies Stan Laurel made for Joe Rock.
  Fact 1.4: The person's directed films include fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
Sentence 2: John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.
  Fact 2.0: John Scott Biddle lived from 1925 to 2008.
  Fact 2.1: John Scott Biddle was a foremost yachting cinematographer.
  Fact 2.2: John Scott Biddle was a lecturer.
  Fact 2.3: John Scott Biddle established a film-making career.
  Fact 2.4: John Scott Biddle's film-making career spanned more than forty years.

Retrieved items:
<item 1>
Charles John Biddle
Charles John Biddle (April 30, 1819 – September 28, 1873) was an American soldier, lawyer, congressman, and newspaper editor.
</item 1>

<item 2>
Millard Webb
Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.

 He directed 20 films between 1920 and 1933.

 His best-known film is the 1926 silent John Barrymore adventure "The Sea Beast" costarring Dolores Costello.

 Webb also directed the early sound Florenz Ziegfeld produced talkie "Glorifying the American Girl" released by Paramount in 1929.

 His active years were from 1916 to 1933.
</item 2>

<item 3>
Charles J. Biddle (aviator)
Major Charles John Biddle (13 May 1890 – 22 March 1972) was an American aviator, attorney, and author.

 He was a flying ace during World War I.

 Postwar, he launched a career in law and wrote his memoirs.
</item 3>

<item 4>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 4>

<item 5>
John Biddle (Michigan politician)
John Biddle (March 2, 1792 – August 25, 1859) was a delegate to the United States Congress from the Michigan Territory.
</item 5>

<item 6>
John Biddle (Unitarian)
John Biddle or Bidle (born Wotton-under-Edge, Gloucestershire, England, 14 January 1615 – died 22 September 1662) was an influential English nontrinitarian, and
Unitarian.

 He is often called "the Father of English Unitarianism".
</item 6>

<item 7>
Biddle family
The American Biddle family of Philadelphia began when William Biddle (1630–1712) and Sarah Kempe (1634–1709), Quakers, emigrated from England to America in
1681, in part, to avoid religious persecution.

 Having acquired extensive rights to over 43000 acre of lands in Quaker West Jersey, they settled first at Burlington.

 Two third generation brothers, William Biddle, 3rd (1698–1756), and John Biddle (1707–1789), moved from Mount Hope (1684) near Bordentown to Philadelphia in
the 1720s and 1730s and constituted the first generation of the Philadelphia Biddle family.
</item 7>

<item 8>
Harry Sweet
Harry Sweet (October 2, 1901 – June 18, 1933) was an American actor, director and screenwriter.

 He appeared in 57 films between 1919 and 1932.

 He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and
fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
</item 8>

<item 9>
John Biddle (United States Army general)
John Biddle (February 2, 1859 – January 18, 1936) was career United States Army officer who became superintendent of the United States Military Academy.
</item 9>

<item 10>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 10>
=== END ITEM id=78577320ce777739 ===

=== ITEM id=0dc5c55a9aadf0b2 ===
Gold sentences and facts:
Sentence 0:  The 381-ton vessel was armed with a 12-pounder 12cwt QF gun, a 20mm Oerlikon cannon, a .303-inch Vickers machine gun, and four Type D depth charges, and was commissioned into the RAN on 11 June 1941.
  Fact 0.0: The vessel weighed 381 tons.
  Fact 0.1: The vessel was armed with a 12-pounder 12cwt QF gun.
  Fact 0.2: The vessel was armed with a 20mm Oerlikon cannon.
  Fact 0.3: The vessel was armed with a .303-inch Vickers machine gun.
  Fact 0.4: The vessel was armed with four Type D depth charges.
  Fact 0.5: The vessel was commissioned into the RAN on 11 June 1941.
Sentence 1:  It was in service from before the First World War until the 1960s, with air-cooled versions of it on many Allied World War I fighter aircraft.
  Fact 1.0: This item was in service from before the First World War until the 1960s.
  Fact 1.1: Air-cooled versions of this item were on many Allied World War I fighter aircraft.

Retrieved items:
<item 1>
AK-630
The AK-630 is a Soviet and Russian fully automatic naval close-in weapon system based on a six-barreled 30 mm rotary cannon.

 In "630", "6" means 6 barrels and "30" means 30 mm.

 It is mounted in an enclosed automatic turret and directed by radar and television detection and tracking.

 The system's primary purpose is defense against anti-ship missiles and other precision guided weapons.

 However it can also be employed against fixed or rotary wing aircraft, ships and other small craft, coastal targets, and floating mines.

 Once operational, this weapon system was rapidly adopted, with up to 8 units installed in every new Soviet warship (from mine-hunters to aircraft carriers),
and hundreds produced in total.
</item 1>

<item 2>
HMAS Narani
HMAS "Narani" was an auxiliary minesweeper operated by the Royal Australian Navy (RAN) during World War II.

 "Narani" was requisitioned from the Illawarra & South Coast Steam Navigation Company as auxiliaries.

 The 381-ton vessel was armed with a 12-pounder 12cwt QF gun, a 20mm Oerlikon cannon, a .303-inch Vickers machine gun, and four Type D depth charges, and was
commissioned into the RAN on 11 June 1941.
</item 2>

<item 3>
HMS Illustrious (87)
Her aircraft attacked several targets in the Japanese-occupied Dutch East Indies over the following year before "Illustrious" was transferred to the newly
formed British Pacific Fleet (BPF).

 The carrier participated in the early stages of the Battle of Okinawa until mechanical defects arising from accumulated battle damage became so severe that she
was ordered home early for repairs in May 1945.
</item 3>

<item 4>
Vickers machine gun
The Vickers machine gun or Vickers gun is a name primarily used to refer to the water-cooled .303 British (7.7 mm) machine gun produced by Vickers Limited,
originally for the British Army.

 The machine gun typically required a six to eight-man team to operate: one fired, one fed the ammunition, the rest helped to carry the weapon, its ammunition,
and spare parts.

 It was in service from before the First World War until the 1960s, with air-cooled versions of it on many Allied World War I fighter aircraft.
</item 4>

<item 5>
SWARM
The SWARM Remote Weapon System (Stabilised Weapon And Reconnaissance Mount) is a fully armored remote weapon system designed and built by the Thales Group in
Glasgow, Scotland.

 The SWARM system consists of two main assemblies: the Gun Processing and Interface Unit (GPIU), which is operated inside the vehicle, and the external Weapon
and Sensor Platform (WASP).

 It can fire a variety of weapons, and utilize multiple sensors.

 On the US Marine Corps' Gladiator Tactical Unmanned Ground Vehicle (TUGV), it is equipped with a 7.62 mm M240 and day/night sensors.
</item 5>

<item 6>
Pershing II
The Pershing II Weapon System was a solid-fueled two-stage ballistic missile designed and built by Martin Marietta to replace the Pershing 1a Field Artillery
Missile System as the United States Army's primary nuclear-capable theater-level weapon.

 The U.S. Army replaced the Pershing 1a with the Pershing II Weapon System in 1983 while the German Air Force retained Pershing 1a until all Pershings were
eliminated in 1991.

 The U.S. Army Missile Command (MICOM) managed the development and improvements while the Field Artillery Branch deployed the systems and developed tactical
doctrine.
</item 6>

<item 7>
HMS Illustrious (87)
HMS "Illustrious" was the lead ship of her class of aircraft carriers built for the Royal Navy before World War II.

 Her first assignment after completion and working up was with the Mediterranean Fleet, in which her aircraft's most notable achievement was sinking one Italian
battleship and badly damaging two others during the Battle of Taranto in late 1940.

 Two months later the carrier was crippled by German dive bombers and was repaired in the United States.

 After sustaining damage on the voyage home in late 1941 by a collision with her sister ship "Formidable" , "Illustrious" was sent to the Indian Ocean in early
1942 to support the invasion of Vichy French Madagascar (Operation Ironclad).

 After returning home in early 1943, the ship was given a lengthy refit and briefly assigned to the Home Fleet.

 She was transferred to Force H for the Battle of Salerno in mid-1943 and then rejoined the Eastern Fleet in the Indian Ocean at the beginning of 1944.
</item 7>

<item 8>
HMS Diomede (D92)
HMS "Diomede" was a "Danae"-class cruiser of the Royal Navy.

 Constructed at Vickers Armstrong, Barrow, she was constructed too late to take part in World War I and was consequently completed at the Royal Dockyard,
Portsmouth.

 Between the wars, she served on the China Station, Pacific waters, East Indies Waters and from 1936 onwards, in reserve.

 In World War II she performed four years of arduous war duty, during which time she captured the German blockade runner "Idarwald" .

 Between 22 July 1942 and 24 September 1943 she was converted to a training ship at Rosyth Dockyard.

 In 1945 she was placed in reserve and scrapped a year later.
</item 8>

<item 9>
List of sunken aircraft carriers
With the advent of heavier-than-air flight, the aircraft carrier has become a decisive weapon at sea.

 In 1911 aircraft began to be successfully launched and landed on ships with the successful flight of a Curtiss Pusher aboard the USS "Pennsylvania".

 The British Royal Navy pioneered the first aircraft carrier as floatplanes, as flying boats under performed compared to traditional land based aircraft.

 The first true aircraft carrier was the HMS "Argus" , launched in late 1917 with a complement of 20 aircraft, a flight deck of 550 ft and 68 ft wide.

 The last aircraft carrier sunk in wartime was the Japanese carrier "Amagi", in Kure Harbour in July 1945.

 The greatest loss of life was the 2,046 killed on the "Akitsu Maru"—a converted passenger liner with a small flight deck, carrying the Imperial Japanese Army's
64th Infantry Regiment.
</item 9>

<item 10>
Weapon system
Weapon System is a United States military term that designated, along with a weapon system number (e.g., WS-110), military experimental (MX) weapons prior to
official naming (e.g., under a military aircraft designation system.

 Preceded by the first Skunk Works program (MX-813 for the Convair XF-92 in 1946), the earliest "WS" designation was the 1954 WS-117L.

 Circa February 1950, an Air Research and Development Command ""study prepared by Maj Gen Gordon P. Saville ... recommended that a 'systems approach' to new
weapons be adopted "[whereby]" development of a weapon "system" required development of support equipment as well as the actual hardware itself.""
</item 10>
=== END ITEM id=0dc5c55a9aadf0b2 ===

=== ITEM id=c76137c01940cd86 ===
Gold sentences and facts:
Sentence 0: Jeffrey Charles "Jeff" Ragsdale ( ) is an American author, documentary filmmaker, actor and stand-up comedian.
  Fact 0.0: Jeffrey Charles "Jeff" Ragsdale is an American author.
  Fact 0.1: Jeffrey Charles "Jeff" Ragsdale is a documentary filmmaker.
  Fact 0.2: Jeffrey Charles "Jeff" Ragsdale is an actor.
  Fact 0.3: Jeffrey Charles "Jeff" Ragsdale is a stand-up comedian.
Sentence 1: Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.
  Fact 1.0: Millard Webb was born on 6 December 1893.
  Fact 1.1: Millard Webb died on 21 April 1935.
  Fact 1.2: Millard Webb was an American screenwriter.
  Fact 1.3: Millard Webb was an American director.

Retrieved items:
<item 1>
Speedy West
Wesley Webb West (January 25, 1924 – November 15, 2003), better known as Speedy West, was an American pedal steel guitarist and record producer.

 He frequently played with Jimmy Bryant, both in their own duo and as part of the regular Capitol Records backing band for Tennessee Ernie Ford and many others.

 He also played on Loretta Lynn's first single.
</item 1>

<item 2>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 2>

<item 3>
Jeff Ragsdale
Jeffrey Charles "Jeff" Ragsdale ( ) is an American author, documentary filmmaker, actor and stand-up comedian.

 In 2011 he posted a flyer in New York City as a "social experiment", stating his phone number and asking people to call him, describing himself as "Jeff, one
lonely guy".

 He was overwhelmed with thousands of calls after photos of the flyer were posted on the internet.

 The experience led to his 2012 book "Jeff, One Lonely Guy", and indirectly to a 2013 pilot episode for a reality television show, "Being Noticed", and a
starring role in the 2014 documentary "Hotline".
</item 3>

<item 4>
Uncle Wiggily
Uncle Wiggily Longears is the main character of a series of children's stories by American author Howard R. Garis.

 He began writing the stories for the "Newark News" in 1910.

 Garis penned an Uncle Wiggily story every day (except Sundays) for more than 30 years, and published 79 books within the author's lifetime.

 According to his obituary in the "Chicago Tribune", a walk in the woods in Verona, New Jersey was his inspiration.

 The books featured work by several illustrators, notably Lansing Campbell.

 Other illustrators of the series included George L. Carlson, Louis Wisa, Elmer Rache, Edward Bloomfield, Lang Campbell and Mary and Wallace Stover.
</item 4>

<item 5>
Richard Yoo
Richard Yoo is an American entrepreneur, the co-founder and former CEO of the web hosting company Rackspace, and the founder and former CEO of web hosting
company ServerBeach.
</item 5>

<item 6>
Howard R. Garis
Howard Roger Garis ((1873--) 25, 1873 – (1962--) 6, 1962 ) was an American author, best known for a series of books, published under his own name, that featured
the character of Uncle Wiggily Longears, an engaging elderly rabbit.

 Garis and his wife were possibly the most prolific children's authors of the early 20th century.

 Many of his books were illustrated by Lansing Campbell.
</item 6>

<item 7>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 7>

<item 8>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 8>

<item 9>
Millard Webb
Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.

 He directed 20 films between 1920 and 1933.

 His best-known film is the 1926 silent John Barrymore adventure "The Sea Beast" costarring Dolores Costello.

 Webb also directed the early sound Florenz Ziegfeld produced talkie "Glorifying the American Girl" released by Paramount in 1929.

 His active years were from 1916 to 1933.
</item 9>

<item 10>
Jeff, One Lonely Guy
Jeff, One Lonely Guy is a 2012 nonfiction book by Jeff Ragsdale.

 It was published on March 20, 2012 by New Harvest.

 Dave Eggers selected the book for inclusion in The Best American Nonrequired Reading 2012, and it was a GQ 2012 "Book of the Year".

 In 2014 Amitava Kumar included portions of "Jeff, One Lonely Guy" in his newly released book, "A Matter of Rats: A Short Biography of Patna".

 Kumar previously interviewed Ragsdale and wrote about him in "The New York Times".
</item 10>
=== END ITEM id=c76137c01940cd86 ===
