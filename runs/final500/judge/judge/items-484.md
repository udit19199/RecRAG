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

This file has 12 items.

=== ITEM id=088216aa1d48d293 ===
Gold sentences and facts:
Sentence 0: Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
  Fact 0.0: Carly Gullickson was born on November 26, 1986.
  Fact 0.1: Carly Gullickson is a former American professional tennis player.
Sentence 1: Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.
  Fact 1.0: Francesca Schiavone was born on 23 June 1980 in Milan.
  Fact 1.1: Francesca Schiavone is an Italian tennis player.
  Fact 1.2: Francesca Schiavone turned professional in 1998.

Retrieved items:
<item 1>
2009 Hansol Korea Open – Doubles
Chuang Chia-jung and Hsieh Su-wei were the defending champion, but Hsieh chose not to participate that year.

 Chuang partnered with Yan Zi, but they lost in the semifinals against Chan Yung-jan and Abigail Spears.

Chan Yung-jan and Abigail Spears won in the final 6–3, 6–4 against Carly Gullickson and Nicole Kriz.
</item 1>

<item 2>
Carly Gullickson
Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
</item 2>

<item 3>
Francesca Schiavone
Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.

 She won the 2010 French Open singles title, becoming the first Italian woman to win a Grand Slam event in singles.

 She was also runner-up at the 2011 French Open.

 Her career high ranking is world No. 4, achieved on 31 January 2011.

 To date, Schiavone is the last one handed-backhand player to win a Grand Slam title on the women's tour.
</item 3>

<item 4>
2007 Kremlin Cup – Women's Doubles
Květa Peschke and Francesca Schiavone were the defending champions, but Schiavone chose not to participate, and only Peschke competed that year.
</item 4>

<item 5>
2005 Challenge Bell – Doubles
Carly Gullickson and María Emilia Salerni were the defending champions, but Gullickson decided not to participate this year.

 Salerni partnered with Marion Bartoli, but withdrew from their semifinal match against Līga Dekmeijere and Ashley Harkleroad.
</item 5>

<item 6>
2010 US Open – Mixed Doubles
Carly Gullickson and Travis Parrott were the defending champions, but lost in the first round to Gisela Dulko and Pablo Cuevas 2–6, 4–6.
</item 6>

<item 7>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 7>

<item 8>
2006 Qatar Ladies Open – Doubles
Alicia Molik and Francesca Schiavone were the defending champions, but Molik did not compete this year.

 Schiavone teamed up with Květa Peschke and lost in querterfinals to Elena Likhovtseva and Vera Zvonareva.
</item 8>

<item 9>
2006 Generali Ladies Linz – Doubles
Gisela Dulko and Květa Peschke were the defending champions, but competed this year with different partners.

 Dulko teamed up with Michaëlla Krajicek and lost in first round to Eleni Daniilidou and Jasmin Wöhr, while Peschke teamed up with Francesca Schiavone and had
to withdrew in their semifinal match (against Corina Morariu and Katarina Srebotnik) as Schiavone had her right knee injured.
</item 9>

<item 10>
1999 Italian Open – Women's Doubles
The 1999 Italian Open – Doubles was the doubles event of the fifty-fifth edition of the tennis tournament played at Rome, Italy, the most prestigious tennis
tournament in Southern Europe.

 It was the fifth WTA Tier I tournament of the year, and part of the European claycourt season.

 Virginia Ruano Pascual and Paola Suárez were the defending champions, but lost to Martina Hingis and Anna Kournikova in the quarterfinals.
</item 10>
=== END ITEM id=088216aa1d48d293 ===

=== ITEM id=55fd6f483a313b82 ===
Gold sentences and facts:
Sentence 0: Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
  Fact 0.0: Ernst Willi Messerschmid is Prof. Dr.
  Fact 0.1: Ernst Willi Messerschmid was born on May 21, 1945.
  Fact 0.2: Ernst Willi Messerschmid is a German physicist.
  Fact 0.3: Ernst Willi Messerschmid is a former astronaut.
Sentence 1: Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
  Fact 1.0: Philippe Perrin was born on January 6, 1963.
  Fact 1.1: Philippe Perrin is a Colonel in the French Air Force.
  Fact 1.2: Philippe Perrin is a French test pilot.
  Fact 1.3: Philippe Perrin is a former CNES astronaut.
  Fact 1.4: Philippe Perrin is a former European Space Agency astronaut.

Retrieved items:
<item 1>
François Englert
He is the recipient of the 2013 Prince of Asturias Award in technical and scientific research, together with Peter Higgs and the CERN.
</item 1>

<item 2>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 2>

<item 3>
Erno S. Daniel
Erno S. Daniel M.D. Ph.D (Born December 15, 1946, Died February 21, 2015) is an internal medicine specialist and author.

 He is of Hungarian descent.

 He has practiced at the Sansum Clinic in Santa Barbara (formerly known as the Santa Barbara Medical Foundation Clinic) since 1978.

 Since that time, he has participated in teaching activities at Santa Barbara's Cottage Hospital, whose internal medicine residency training program is
affiliated with University of Southern California School of Medicine.

 Because of his activities, for a period of time Dr. Daniel held an appointment as Associate Clinical Professor of Medicine at USC.

 He also participated as a lecturer for the Introduction to Medicine course offered by the University of California at Santa Barbara.
</item 3>

<item 4>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 4>

<item 5>
Serge Haroche
Serge Haroche (born 11 September 1944) is a French physicist who was awarded the 2012 Nobel Prize for Physics jointly with David J. Wineland for
"ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems", a study of the particle of light, the photon.

 This and his other works developed laser spectroscopy.

 Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.

 In 1971 he defended his doctoral thesis in physics at the University of Paris VI, his research has been conducted under the direction of Claude
Cohen-Tannoudji.
</item 5>

<item 6>
Philippe Perrin
Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
</item 6>

<item 7>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 7>

<item 8>
Ekhard Salje: Cambridge mineralogist and former Earth Sciences head. Peter Doherty: Australian veterinary immunologist; 1995 Lasker, 1996 medicine Nobel with
Zinkernagel, 1997 Australian of the Year. Antje Boetius: German geomicrobiologist; 2009 Leibniz laureate, studies seabed methane-oxidizing microbes and climate.
Brian Josephson: Welsh physicist; 1973 physics Nobel for the Joseph
</item 8>

<item 9>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 9>

<item 10>
Philippe Perrin (artist)
Philippe Perrin, (La Tronche, 10 August 1964) is a French artist and contemporary photographer who lives and works in Paris.
</item 10>
=== END ITEM id=55fd6f483a313b82 ===

=== ITEM id=f8ad85ce9364846a ===
Gold sentences and facts:
Sentence 0: "I'll Keep the Lovelight Burning" is a popular song written in 1942 by Harry Tobias, Nick Kenny, and Harold Levey, popularized by Patti Page in 1949.
  Fact 0.0: "I'll Keep the Lovelight Burning" is a popular song.
  Fact 0.1: "I'll Keep the Lovelight Burning" was written in 1942.
  Fact 0.2: "I'll Keep the Lovelight Burning" was written by Harry Tobias.
  Fact 0.3: "I'll Keep the Lovelight Burning" was written by Nick Kenny.
  Fact 0.4: "I'll Keep the Lovelight Burning" was written by Harold Levey.
  Fact 0.5: "I'll Keep the Lovelight Burning" was popularized by Patti Page.
  Fact 0.6: Patti Page popularized "I'll Keep the Lovelight Burning" in 1949.
Sentence 1: Clara Ann Fowler (November 8, 1927 – January 1, 2013), known by her professional name Patti Page, was an American singer of traditional pop music and country music.
  Fact 1.0: Clara Ann Fowler's date range is November 8, 1927 to January 1, 2013.
  Fact 1.1: Clara Ann Fowler is known by the professional name Patti Page.
  Fact 1.2: Patti Page was an American singer.
  Fact 1.3: Patti Page was a singer of traditional pop music.
  Fact 1.4: Patti Page was a singer of country music.

Retrieved items:
<item 1>
You Keep Me Hangin' On
"You Keep Me Hangin' On" is a 1966 song written and composed by Holland–Dozier–Holland.

 It first became a popular "Billboard" Hot 100 number one hit for the American Motown group The Supremes in late 1966.

 The rock band Vanilla Fudge covered the song a year later and had a top ten hit with their version.

 British pop singer Kim Wilde covered "You Keep Me Hangin' On" in 1986, bumping it back to number one on the "Billboard" Hot 100 in June 1987.

 The single reached number one by two different musical acts in America.

 In the first 32 years of the "Billboard" Hot 100 rock era, “You Keep Me Hangin' On” became one of only six songs to achieve this feat.

 In 1996, country music singer Reba McEntire's version reached number 2 on the US "Billboard" Hot Dance Club Play chart.
</item 1>

<item 2>
I'll Keep the Lovelight Burning
"I'll Keep the Lovelight Burning" is a popular song written in 1942 by Harry Tobias, Nick Kenny, and Harold Levey, popularized by Patti Page in 1949.

 Louis Armstrong also covered the song in 1949.
</item 2>

<item 3>
Willie and the Hand Jive
"Willie and the Hand Jive" is a song written by Johnny Otis and originally released as a single in 1958 by Johnny Otis Show, reaching #9 on the "Billboard" Hot
100 chart and #5 on the Billboard R&B chart.

 The song has a Bo Diddley beat and was partly inspired by the music sung by a chain gang Otis heard while he was touring.

 The lyrics are about a man who became famous for doing a dance with his hands, but the song has been accused of glorifying masturbation.

 It has since been covered by numerous artists, including The Strangeloves, Eric Clapton, Cliff Richard, Kim Carnes, George Thorogood and The Grateful Dead.

 Clapton's 1974 version was also released as a single and also reached the Billboard Top 40, peaking at #26.

 Thorogood's 1985 version reached #25 on the Hot Mainstream Rock Tracks chart.
</item 3>

<item 4>
Patti Page
Clara Ann Fowler (November 8, 1927 – January 1, 2013), known by her professional name Patti Page, was an American singer of traditional pop music and country
music.

 She was the top-charting female vocalist and best-selling female artist of the 1950s, selling over 100 million records during a six-decade long career.

 She was often introduced as "the Singin' Rage, Miss Patti Page".

 New York WNEW disc-jockey William B. Williams introduced her as "A Page in my life called Patti".
</item 4>

<item 5>
Live in Old Smokey
Live in Old Smokey is a live CD by English singer/songwriter Linda Lewis recorded at Ronnie Scott's in London and issued in 2006.
</item 5>

<item 6>
Keep This Fire Burning
"Keep This Fire Burning" is a song by Swedish pop singer Robyn, released as the first single from her third album "Don't Stop the Music".

 It was released in Sweden on September 21, 2002, where it became her highest charting single since 1995's "Do You Really Want Me (Show Respect)".

 The song was also released as a single in Australia under the name "By Your Side", due to the Australian bush fires which were happening at the time.

 In 2008, a re-recorded version of the song appeared on the special edition of Robyn's eponymous album.
</item 6>

<item 7>
Elton John
Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.

 He has worked with lyricist Bernie Taupin as his songwriting partner since 1967; they have collaborated on more than 30 albums to date.

 In his five-decade career Elton John has sold more than 300million records, making him one of the best-selling music artists in the world.

 He has more than fifty Top 40 hits, including seven consecutive No. 1 US albums, 58 "Billboard" Top 40 singles, 27 Top 10, four No. 2 and nine No. 1.

 For 31 consecutive years (1970–2000) he had at least one song in the "Billboard" Hot 100.

 His tribute single, re-penned in dedication to the late Princess Diana, "Candle in the Wind 1997" sold over 33million copies worldwide and is the best-selling
single in the history of the UK and US singles charts.

 He has also composed music, produced records, and has occasionally acted in films.
</item 7>

<item 8>
For You (Ricky Nelson song)
"For You" is a song written by Joe Burke and Al Dubin in 1930.

 Dubin wrote the songs for the Broadway show "42nd Street".

 The Glen Gray Orchestra recorded it with Kenny Sargent doing the vocals.

 Perry Como recorded it in November 1947, releasing the song in 1948.

 Dean Martin recorded it for an album in 1957.

 Timi Yuro also recorded the song.

 It was released on her LP "Timi Yuro" (Liberty LRP-3208 (mono) and Liberty LST-7208 (stereo)) in 1961.
</item 8>

<item 9>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 9>

<item 10>
Cecil Gant
Cecil Gant (April 4, 1913 – February 4, 1951) was an American blues singer, songwriter and pianist, whose recordings of both ballads and "fiery piano rockers"
were successful in the mid- and late 1940s, and influenced the early development of rock and roll.

 His biggest hit was the 1944 ballad, "I Wonder".
</item 10>
=== END ITEM id=f8ad85ce9364846a ===

=== ITEM id=62625fd24a57054b ===
Gold sentences and facts:
Sentence 0:  Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.
  Fact 0.0: Qionghai is called a city.
  Fact 0.1: Qionghai refers to a large land area in Hainan.
  Fact 0.2: The area of Qionghai in Hainan was once a county.
Sentence 1:  It has an area of 1692 km2 and in 2010, it had a population of 483,217.
  Fact 1.0: Qionghai has an area of 1692 km2.
  Fact 1.1: Qionghai had a population of 483,217 in 2010.
Sentence 2: Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.
  Fact 2.0: Suining has the Sichuanese Pinyin Xunin.
  Fact 2.1: Suining is a prefecture-level city.
  Fact 2.2: Suining is in eastern Sichuan province.
  Fact 2.3: Sichuan province is in Southwest China.
Sentence 3:  In 2002, Suining had a population of 658,798.
  Fact 3.0: Suining had a population of 658,798 in 2002.

Retrieved items:
<item 1>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 1>

<item 2>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 2>

<item 3>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
</item 3>

<item 4>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 4>

<item 5>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 5>

<item 6>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 6>

<item 7>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 7>

<item 8>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 8>

<item 9>
Qionghai
Qionghai () is one of the seven county-level cities of Hainan province, China.

 Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.

 Within this area is the main city, Qionghai City.

 It is located in the east of the island at the mouth of the Wanquan River, 86 km from the provincial capital of Haikou.

 It has an area of 1692 km2 and in 2010, it had a population of 483,217.
</item 9>

<item 10>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 10>
=== END ITEM id=62625fd24a57054b ===

=== ITEM id=4331c674fa3d3ce6 ===
Gold sentences and facts:
Sentence 0:  It plays during the opening sequence and was composed by Danny Elfman in 1989, after series creator Matt Groening approached him requesting a retro-style theme.
  Fact 0.0: The referenced piece plays during the opening sequence.
  Fact 0.1: Danny Elfman composed the piece in 1989.
  Fact 0.2: Danny Elfman composed the piece after Matt Groening approached him.
  Fact 0.3: Matt Groening is the series creator.
  Fact 0.4: Matt Groening requested a retro-style theme.
Sentence 1:  Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's Day Off".
  Fact 1.0: Clausen has scored or orchestrated music for more than 30 films and television shows.
  Fact 1.1: Clausen has scored or orchestrated music for 'Moonlighting'.
  Fact 1.2: Clausen has scored or orchestrated music for 'The Naked Gun'.
  Fact 1.3: Clausen has scored or orchestrated music for 'ALF'.
  Fact 1.4: Clausen has scored or orchestrated music for 'Ferris Bueller's Day Off'.

Retrieved items:
<item 1>
Disney's Adventures of the Gummi Bears
The show is well-remembered for its theme music, written by Michael and Patty Silversher and creation of "gummiberry juice" which was a type of magic potion,
granting abilities, which allowed them to bounce away from their hunters.

 The theme song was performed by Joseph Williams, son of composer John Williams and one-time lead singer of Toto.
</item 1>

<item 2>
Angelo Badalamenti
Angelo Badalamenti (born March 22, 1937) is an American composer, best known for his work scoring films for director David Lynch, notably "Blue Velvet", the
"Twin Peaks" saga (1990–1992, 2017), "The Straight Story" and "Mulholland Drive".

 Badalamenti received the 1990 Grammy Award for Best Pop Instrumental Performance for his ""Twin Peaks" Theme", and has received a Lifetime Achievement Award
from the World Soundtrack Awards and the Henry Mancini Award from the American Society of Composers, Authors and Publishers.
</item 2>

<item 3>
Michelangelo Sosnowitz
Michelangelo Sosnowitz (born August 21, 1973) is an American music composer.

 His works include ballet, musical theater, electronic and popular music, as well as commercial, television and film.

 His feature film scores include A Novel Romance, winner Best Film at the NYC International Independent Film Festival, the Dan Fogler directorial debut
Hysterical Psycho and the Deborah Kampmeier film Split.

 His original ballet 'Petrouchka' won Best Choreography at NYMF 2010.

 He wrote the opening theme music to The History Channel show Battles BC and original theme music for ENTV, Deadline Hollywood BlackTreeMedia and other PMC
programming.

 He also wrote the score and theme music to the Colin Quinn Web series COP SHOW.

 Original Off-Broadway musicals include Crazy Head Space and Love in the MIddle Ages.
</item 3>

<item 4>
Batman (score)
Batman: Original Motion Picture Score is the score album for the 1989 film "Batman" by Danny Elfman.

 According to the "Batman" DVD Special Edition, Elfman said that producer Jon Peters was not sure about him as a composer until Tim Burton made him play the
main titles.

 Elfman admitted he was stunned when Peters announced that the score would be released on its own album, as releasing a separate score album for a film was
something that was rarely done in the 1980s.

 Elfman's "The Batman Theme" went on to become an iconic piece.

 It served as the basis for the theme music of "", which premiered in 1992, although this was later changed.

 Some parts of the Elfman score are also heard in "", "" and "".

 Parts are also played in the queue, and on the station platform of Batman the Ride at various Six Flags theme parks.
</item 4>

<item 5>
The Simpsons Theme
"The Simpsons" Theme", also referred to as "The Simpsons" Main Title Theme" in album releases, is the theme music of the animated television series "The
Simpsons".

 It plays during the opening sequence and was composed by Danny Elfman in 1989, after series creator Matt Groening approached him requesting a retro-style
theme.

 The piece, which took 3 days, 2 hours, 48 minutes, and 19 seconds to create, has been noted by Elfman as the most popular of his career.

 The theme, as used for the opening sequence, was re-arranged during season 2, and the current arrangement by Alf Clausen was introduced at the beginning of the
third season.
</item 5>

<item 6>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
</item 6>

<item 7>
Alf Clausen
Alf Heiberg Clausen (born March 28, 1941) is an American film and television composer.

 He is best known for his work scoring many episodes of "The Simpsons", of which he had been the sole composer between 1990 and 2017.

 Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's
Day Off".
</item 7>

<item 8>
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 8>

<item 9>
James Dooley (composer)
James Michael "Jim" Dooley (born August 22, 1976 in New York City, New York) is an American film score composer.

 Dooley studied music at New York University, majoring in music composition.

 After finishing the university he moved to Los Angeles, where he studied music with prolific film score composers Christopher Young, Elmer Bernstein and
Leonard Rosenman.

 In 1999, he started working for Hans Zimmer as his chief technical assistant.

 He works in Santa Monica, in Hans Zimmer's film music studio Remote Control Productions (formerly "Media Ventures").

 He composed, arranged, and orchestrated music for films like "" and "The Da Vinci Code".

 He also composed music for "inFAMOUS 2", the Epic Mickey series and worked with Celldweller and Tarja Turunen.

 He released his debut album, "Veiled Nation", in 2013.
</item 9>

<item 10>
Diwa de Leon
Diwa de Leon is a composer, arranger and musician based in Manila, Philippines.

 He makes music mainly for television shows, films, and stage productions.

 His most notable work is his arrangement of GMA Network's "Survivor Philippines" theme music.

 His songs included in "Emir".

 He won the Cinema One Originals Best Musical Score award for his work on the film "Kolorete" and Best Music Score for "Kamera Obskura" at the 8th Cinemalaya
Independent Film Festival.

 He is also a founding member of the world music group Makiling.

 His first mainstream film project is Mamarazzi produced by Regal Films.
</item 10>
=== END ITEM id=4331c674fa3d3ce6 ===

=== ITEM id=a8a77f4dc03612f4 ===
Gold sentences and facts:
Sentence 0: Gasland is a 2010 American documentary written and directed by Josh Fox.
  Fact 0.0: Gasland is a documentary.
  Fact 0.1: Gasland is from 2010.
  Fact 0.2: Gasland is American.
  Fact 0.3: Gasland was written by Josh Fox.
  Fact 0.4: Gasland was directed by Josh Fox.
Sentence 1: To Shoot an Elephant is a 2009 documentary film about the 2008-2009 Gaza War directed by Alberto Arce and Mohammad Rujailahk.
  Fact 1.0: To Shoot an Elephant is a 2009 documentary film.
  Fact 1.1: To Shoot an Elephant is about the 2008-2009 Gaza War.
  Fact 1.2: To Shoot an Elephant was directed by Alberto Arce.
  Fact 1.3: To Shoot an Elephant was directed by Mohammad Rujailahk.

Retrieved items:
<item 1>
To Shoot an Elephant
To Shoot an Elephant is a 2009 documentary film about the 2008-2009 Gaza War directed by Alberto Arce and Mohammad Rujailahk.
</item 1>

<item 2>
Hunting Elephants
Hunting Elephants (Hebrew: לצוד פילים‎ ‎ , "Latzud Pilim", "lit."

 To Hunt Elephants) is a 2013 Israeli crime comedy film directed by Reshef Levi.

 It stars, among others, Sasson Gabai, Moni Moshonov and Patrick Stewart.

 It was released to theaters in Israel on July 4, 2013.
</item 2>

<item 3>
Olympia (1938 film)
Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium
in Berlin, Germany.

 The film was released in two parts: "Olympia 1.

 Teil — Fest der Völker" (Festival of Nations) and "Olympia 2.

 Teil — Fest der Schönheit" (Festival of Beauty).

 It was the first documentary feature film of the Olympic Games ever made.

 Many advanced motion picture techniques, which later became industry standards but which were groundbreaking at the time, were employed —including unusual
camera angles, smash cuts, extreme close-ups, placing tracking shot rails within the bleachers, and the like.

 The techniques employed are almost universally admired, but the film is controversial due to its political context.

 Nevertheless, the film appears on many lists of the greatest films of all time, including "Time" magazine's "All-Time 100 Movies."
</item 3>

<item 4>
Elephant (album)
Elephant is the fourth album by the American alternative rock duo The White Stripes.

 Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the
Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album
charts.
</item 4>

<item 5>
Gasland
Gasland is a 2010 American documentary written and directed by Josh Fox.

 Nominated for an Academy Award for Best Documentary in 2011, the film focuses on communities in the United States affected by natural gas drilling and,
specifically, a method of horizontal drilling into shale formations known as hydraulic fracturing.

 The film was a key mobilizer for the anti-fracking movement, and "brought the term 'hydraulic fracturing' into the nation's living rooms" according to "The New
York Times".

 The film was nominated for the Academy Award for Best Documentary Feature.
</item 5>

<item 6>
South of the Border (2009 film)
South of the Border is a 2009 American documentary film directed by Oliver Stone.

 The documentary premiered at the 2009 Venice Film Festival.

 Writer for the project Tariq Ali calls the documentary "a political road movie".

 Stone stated that he hopes the film will help people better understand a leader who is wrongly ridiculed "as a strongman, as a buffoon, as a clown."
</item 6>

<item 7>
Baghdad ER
Baghdad ER is a documentary released by HBO on May 21, 2006.

 It shows the Iraq war from the perspective of a military hospital in Baghdad.

 It has some relatively disturbing scenes in it (e.g. amputations), therefore the U.S. Army is officially warning that military personnel watching it could
experience symptoms of post-traumatic stress disorder (PTSD).
</item 7>

<item 8>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 8>

<item 9>
The Last Lions
The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.

 It was shot at the Okavango Delta in Botswana.

 The film premiered at the Palm Springs International Film Festival in January 2011 and was released in select theaters on February 18, 2011.

 The film follows in the tradition of other National Geographic big cat films, such as "India: Land of the Tiger" and "Eye of the Leopard".
</item 9>

<item 10>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 10>
=== END ITEM id=a8a77f4dc03612f4 ===

=== ITEM id=85dfa60412f47be3 ===
Gold sentences and facts:
Sentence 0:  The Park is home to the first flume ride in Ireland (Pirates Plunge).
  Fact 0.0: The Park is home to the first flume ride in Ireland.
  Fact 0.1: The first flume ride in Ireland is named Pirates Plunge.
Sentence 1: Grace O'Malley (c. 1530 – c. 1603; also Gráinne O'Malley, Irish: "Gráinne Ní Mháille" ) was lord of the Ó Máille dynasty in the west of Ireland, following in the footsteps of her father Eoghan Dubhdara Ó Máille.
  Fact 1.0: Grace O'Malley lived from about 1530 to about 1603.
  Fact 1.1: Grace O'Malley is also called Gráinne O'Malley.
  Fact 1.2: Grace O'Malley's Irish name is Gráinne Ní Mháille.
  Fact 1.3: Grace O'Malley was lord of the Ó Máille dynasty in the west of Ireland.
  Fact 1.4: Grace O'Malley followed in the footsteps of her father, Eoghan Dubhdara Ó Máille.

Retrieved items:
<item 1>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 1>

<item 2>
Treasure Island Resort &amp; Casino
An expansion that was completed in fall 2008 included 230 new hotel rooms, 30,000-square-foot event center and a bowling center complete with an arcade area.

 In 2015, Tado Steakhouse was constructed, Tradewinds Buffet was remodeled and the water park & spa construction began.

 The Lagoon and Wave Spa opened February 9, 2016.
</item 2>

<item 3>
Reef Hawaiian Pro
The Reef Hawaiian Pro is a surfing event held annually on the North Shore, Oahu, Hawaii.

 It is currently the first leg of the Triple Crown of Surfing.

 Surfers compete for a share of $135,000 (men) or $35,000 (women) prize money.

 Since it carries a 6-star rating, the event is critical for those who want to qualify for the ASP World Tour.
</item 3>

<item 4>
Team Hole in the Wall
Team Hole in the Wall is an athletic fundraising initiative managed by the Hole in the Wall Gang Camp.

 Amateur athletes join up to raise a specified amount of funds and receive entrance to a marathon, bike ride or other athletic event in support of seriously ill
children served by The Hole in the Wall Gang Camp or one of its associated camps around the world.

 Established in 2005, Team Hole in the Wall offers entrance into more than two dozen marathons and cycling events, including the New York City Marathon, Boston
Marathon and AngelRide cycling event in Connecticut.

 New Canaan resident and Newman's Own Vice President of Marketing Michael “Mike” Havard ran in the ING New York City Marathon prior to 2004, but in that year he
decided to use his participation to raise funds for The Hole in the Wall Gang Camp, raising nearly $40,000.

 The next year, he served as captain of the first Team Hole in the Wall team of 40 runners, raising $150,000.

 The number of Team Hole in the Wall members and events has risen each year since.

 In 2010, more than 1,800 Team Hole in the Wall members are expected to participate in 20 athletic events.
</item 4>

<item 5>
White Water Landing (Dorney Park)
White Water Landing is a Shoot-the-Chutes ride at Dorney Park & Wildwater Kingdom amusement park in Allentown, Pennsylvania.

 It was built in 1992, 10 years after a sister park named Cedar Point Built a defunct flume using the same name: White Water Landing (Cedar Point).

 This ride is identical to Snake River Falls at Cedar Point in Sandusky, Ohio.

 It is located in the upper side near Hydra the Revenge.
</item 5>

<item 6>
Pirate Adventure Park
Pirate Adventure Park is a small theme park located beside Westport House in Westport, Ireland.

 The Park is home to the first flume ride in Ireland (Pirates Plunge).

 The Park is Pirate themed due to Wesport House's association with Grace O'Malley.

 The Park was developed by The 11th Marquess of Sligo on the grounds of Westport House.
</item 6>

<item 7>
Luna Park, Schenectady
Roughly seven decades before the Skycoaster rides that now dot various United States amusement parks, Luna/Rexford Park featured an aerial swing ride.
</item 7>

<item 8>
Christchurch Adventure Park
Christchurch Adventure Park is an Adventure park in the Port Hills of Christchurch, New Zealand.

 Built by the Canadian company Select Evolution, it had its opening function on 16 December 2016.

 The 1500 holders of special passes could ride from 17 December, while the park opened to the public on 21 December.

 According to the developer, the 358 ha park with a 1.8 km chairlift and initially 50 km of downhill tracks is the largest facility of its kind in the Southern
Hemisphere.

 The park closed on 13 February as a precaution due to a nearby fire and two days later, most of the park's tree cover had been destroyed in a large wildfire.

 The lift and ziplines have suffered significant damage, and the park's condition is much worse than the operator initially thought.

 No reopening date has been set, and "it won't be a quick fix".
</item 8>

<item 9>
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
</item 9>

<item 10>
Glenwood Caverns Adventure Park
Glenwood Caverns Adventure Park is an adventure park located above Glenwood Springs, Colorado, about 160 miles west of Denver.

 Prior to 2003, only cave tours were available until a major expansion took place.

 The park is unique because it sits at an altitude of 7,100 feet on a mountain above Glenwood.

 Today, the park features numerous attractions in addition to the cave tours.
</item 10>
=== END ITEM id=85dfa60412f47be3 ===

=== ITEM id=675dccfbac867be2 ===
Gold sentences and facts:
Sentence 0:  It flows in a west to east direction from the western part of the state in the Everglades past the Miami International Airport to a salinity control center near the Miami River.
  Fact 0.0: It flows in a west to east direction.
  Fact 0.1: It flows from the western part of the state in the Everglades.
  Fact 0.2: It flows past the Miami International Airport.
  Fact 0.3: It flows to a salinity control center near the Miami River.
Sentence 1: The Genesee River is a tributary of Lake Ontario flowing northward through the Twin Tiers of Pennsylvania and New York in the United States.
  Fact 1.0: The Genesee River is a tributary of Lake Ontario.
  Fact 1.1: The Genesee River flows northward.
  Fact 1.2: The Genesee River flows through the Twin Tiers of Pennsylvania.
  Fact 1.3: The Genesee River flows through the Twin Tiers of New York in the United States.

Retrieved items:
<item 1>
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 1>

<item 2>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 2>

<item 3>
Tamiami Canal
The Tamiami Canal or C-4 Canal, is a canal located in southern Florida in the United States.

 It flows in a west to east direction from the western part of the state in the Everglades past the Miami International Airport to a salinity control center
near the Miami River.

 It averages 8 ft in depth and is over 100 ft wide in some areas.
</item 3>

<item 4>
Genesee Valley Park
Genesee Valley Park is located in the south side of Rochester, New York along the shores of the Genesee River.

 The New York State Barge Canal (the currently in-use portion of the Erie Canal) crosses the Genesee River within the park.

 The University of Rochester is located near the park's north entrance.
</item 4>

<item 5>
Oatka Creek
Oatka Creek ( ) is the third longest tributary of the Genesee River, located entirely in the Western New York region of the U.S. state of New York.

 From southern Wyoming County, it flows 58 mi to the Genesee near Scottsville, draining an area of 215 sqmi that includes all or part of 23 towns and villages
in Wyoming, Genesee, Livingston and Monroe counties as well.

 Its name means "leaving the highlands" or "approaching an opening" in Seneca.
</item 5>

<item 6>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 6>

<item 7>
Broad Street Bridge (Rochester, New York)
Erie Canal: Second Genesee Aqueduct, also known as the Broad Street Aqueduct or Broad Street Bridge, is a historic stone aqueduct located at Rochester in Monroe
County, New York.

 It was constructed in 1836-1842 and originally carried the Erie Canal over the Genesee River.

 The overall length of the aqueduct including the wings and abutments is 800 ft .

 The aqueduct is 70 ft wide and has massive parapets on either side.

 It is one of four major aqueducts in the mid-19th century Erie Canal system.

 In 1927, a roadbed was added to carry automobile traffic and named Broad Street.

 It also carried a part of the Rochester Subway.
</item 7>

<item 8>
Flagami
The Flagami is a neighborhood in Miami, Florida, United States, roughly defined as south and east of the Tamiami Canal, north of the Tamiami Trail (US 41/South
Eighth Street), and west of Red Road (SR 959/West 57th Avenue), bisected by Flagler Street.
</item 8>

<item 9>
Black Creek (Genesee River)
Black Creek is a tributary of the Genesee River that runs for 46 mi in western New York, United States.

 The creek begins in Middlebury in Wyoming County and runs north for roughly the first half of its course and east for the other half, eventually joining the
Genesee River in Chili, Monroe County.

 Its drainage area spans around 202 mi2 , which is largely rural and agricultural.

 As of 2000, 40,000 residents lived in the creek's drainage area.
</item 9>

<item 10>
Genesee River
The Genesee River is a tributary of Lake Ontario flowing northward through the Twin Tiers of Pennsylvania and New York in the United States.
</item 10>
=== END ITEM id=675dccfbac867be2 ===

=== ITEM id=141b7fad2ee1506c ===
Gold sentences and facts:
Sentence 0:  All matches were played at HC Davos's home arena, Eisstadion Davos.
  Fact 0.0: All matches were played at the home arena of HC Davos.
  Fact 0.1: The home arena of HC Davos is Eisstadion Davos.
Sentence 1: Hockey Club Davos is a professional Swiss ice hockey club based in Davos, Switzerland.
  Fact 1.0: Hockey Club Davos is a professional Swiss ice hockey club.
  Fact 1.1: Hockey Club Davos is based in Davos, Switzerland.

Retrieved items:
<item 1>
The 2015 Macedonian Supercup ended 1–1 after extra time; Vardar beat Rabotnički 4–3 on penalties. American defenseman Zach Bogosian, drafted third overall in
2008, played for Buffalo and represented the U.S. in 2009. Davos’s Vaillant Arena hosts HC Davos and the annual Spengler Cup, founded in 1923. The 1992 final:
Canada defeated Färjestad
</item 1>

<item 2>
1992 Stanley Cup Finals
The 1992 Stanley Cup Finals NHL championship series was contested by the Prince of Wales Conference and defending Stanley Cup champion Pittsburgh Penguins and
the Clarence Campbell Conference champion Chicago Blackhawks.

 The Blackhawks were appearing in their first Finals since .

 After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in
four games, and win their second consecutive and second overall Stanley Cup.

 It was the 99th year of the Stanley Cup, and the first to extend into the month of June.

 It was the last final for Chicago Stadium as it closed in 1994.
</item 2>

<item 3>
1992 Spengler Cup
The 1992 Spengler Cup was held in Davos, Switzerland, between December 26, 1992, and December 31, 1992.

 All matches were played at HC Davos's home arena, Eisstadion Davos.

 The final was won 6-5 by Team Canada over Färjestad BK.
</item 3>

<item 4>
Spengler Cup
The Spengler Cup is an annual invitational ice hockey tournament held in Davos, Switzerland.

 First held in 1923, the Spengler Cup is often cited as the oldest invitational ice hockey tournament in the world.

 The event is hosted by the Swiss team HC Davos and played each year in Davos, Switzerland, between Christmas (December 25) and New Year's Day.

 Currently, all games are held at Vaillant Arena.
</item 4>

<item 5>
1982 Donnay Indoor Championships
The 1982 Donnay Indoor Championships was a men's tennis tournament played on indoor carpet courts in Brussels in Belgium the event was part of the 1982 Volvo
Grand Prix.

 The tournament was held from 8 March through 14 March 1982.

 Third-seeded Vitas Gerulaitis won the singles title.
</item 5>

<item 6>
HC Davos
Hockey Club Davos is a professional Swiss ice hockey club based in Davos, Switzerland.

 Davos plays in the National League (NL).

 The team is usually a strong force in the league and often bolster their roster with Swiss national team players and players who once played for National
Hockey League teams.

 HC Davos also has hosted the annual Spengler Cup tournament, an invitational competition, since 1923.
</item 6>

<item 7>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 7>

<item 8>
2016 Spengler Cup
The 90th Spengler Cup was an ice hockey competition being held in Davos, Switzerland between 26 and 31 December 2016.

 All matches were played at HC Davos's home known as Vaillant Arena.

 Six competing teams were split into two groups of three (in the round-robin series).

 The two groups, named Torriani and Cattini, are named after legendary Swiss hockey players Richard 'Bibi' Torriani and the Cattini brothers, Hans and
Ferdinand.
</item 8>

<item 9>
Vaillant Arena
Vaillant Arena (Eisstadion Davos) is an arena in Davos, Switzerland.

 It is primarily used for ice hockey and is the home arena of HC Davos.

 It holds 7,080 people, of which 3,280 are seated.

 Every year the Spengler Cup is played in this arena.
</item 9>

<item 10>
2015 Spengler Cup
The 89th Spengler Cup was an ice hockey competition held in Davos, Switzerland between 26 and 31 December 2015.

 All matches were played at HC Davos's home known as Vaillant Arena.

 Six competing teams were split into two groups of three (in the round-robin series).

 The two groups, named Torriani and Cattini, are named after legendary Swiss hockey players Richard 'Bibi' Torriani and the Cattini brothers, Hans and
Ferdinand.
</item 10>
=== END ITEM id=141b7fad2ee1506c ===

=== ITEM id=683d3426e3656f5f ===
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
Flaming Doctor Pepper
A Flaming Doctor Pepper is a flaming cocktail said to taste like the soft drink Dr Pepper, although Dr Pepper is not one of its ingredients.
</item 1>

<item 2>
Volcano bowl
Volcano bowls are ceramic drinkware originally associated with mid-20th century American tiki bars and tropical-themed restaurants.

 Drinks served in volcano bowls are typically rum-based, mixed with tropical fruit juices and other liquors such as brandy, vodka, and triple sec, and garnished
with fruit.

 The Flaming Volcano cocktail is especially associated with this drinkware.
</item 2>

<item 3>
Flaming volcano
The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.

 It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about
32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.

 The cone includes a second, much smaller central pit or "crater" which is partially filled with rum or other flammable high-alcohol liquor.

 The "crater" liquor is then ignited, creating a mild volcanic ambience with its central blue flame.
</item 3>

<item 4>
Hotlum Glacier
The Hotlum Glacier is a glacier situated on the northeast flank of Mount Shasta, in the US state of California.

 It is the largest and most voluminous glacier in California, although not as thick or long as the nearby Whitney Glacier.

 The Hotlum Glacier flows from a large cirque on the northeast side of Mount Shasta's main summit below the Hotlum Headwall at roughly 13600 ft .

 It flows northeastward down the steep slope, forming three lobes which terminate near 10400 ft .
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
Flaming beverage
Flaming beverages include cocktails and other mixed drinks that contain flammable, high-proof alcohol, which is ignited prior to consumption.

 The alcohol may be an integral part of the drink, or it may be floated as a thin layer across the top of the drink.

 The flames are mostly for dramatic flair.

 However, in combination with certain ingredients, the flavor of the drink is altered.

 Some flavors are enhanced, and it may impart a toasted flavor to some drinks.
</item 6>

<item 7>
Man-eating tree
Man-eating tree can refer to any of the various legendary carnivorous plants large enough to kill and consume a person or other large animal.

 The carnivorous plant with the largest known traps is probably Nepenthes rajah, which produces pitchers up to 41 cm tall with a volume up to 3.5 l .

 The pitcher of this species are designed to trap arthropods.

 However, the same bait may also attract rodents like the summit rat ("Rattus baluensis") and the Mountain treeshrew ("Tupaia montana").

 Only very rarely will the rodents fall into the large pitchers of this species.

 Other large carnivorous plants that have similar properties include Nepenthes robcantleyi and Nepenthes attenboroughi.
</item 7>

<item 8>
Flaming Moe's
"Flaming Moe's" is the tenth episode of "The Simpsons"' third season.

 It first aired on the Fox network in the United States on November 21, 1991.

 In the episode, Homer tells Moe Szyslak of a secret alcoholic cocktail that includes cough medicine and fire that he calls "Flaming Homer".

 Moe steals the recipe from Homer, renames the drink the "Flaming Moe" and begins selling it at his tavern.

 The drink is a success and boosts business and patronage, but Homer is angry at Moe for his betrayal, and seeks revenge.
</item 8>

<item 9>
Apple-Kneel
The Apple-Kneel is a cocktail.

 It consists of three shots, two Sourz Apple and one Cherry Brandy.

 Each shot is taken from an average shot glass which is around 1.5 ounces (roughly 45 ml).

 Firstly, you put two shots of Sourz Apple into a glass and then put in one shot of Cherry Brandy in and mix.

 It is customary to sip this cocktail.

 It can be served 'neat' or 'on the rocks'.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=683d3426e3656f5f ===

=== ITEM id=20240dca1da39aa8 ===
Gold sentences and facts:
Sentence 0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.
  Fact 0.0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music.
  Fact 0.1: More Hot Rocks was released by ABKCO Records.
  Fact 0.2: ABKCO Records is run by former manager Allen Klein.
  Fact 0.3: Allen Klein usurped control of the Rolling Stones' Decca/London material in 1970.
  Fact 0.4: More Hot Rocks was released after the Rolling Stones departed from Decca.
  Fact 0.5: More Hot Rocks was released after the Rolling Stones departed from Allen Klein.
Sentence 1: Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording artists.
  Fact 1.0: Allen Klein was born December 18, 1931.
  Fact 1.1: Allen Klein died July 4, 2009.
  Fact 1.2: Allen Klein was an American businessman.
  Fact 1.3: Allen Klein was a music publisher.
  Fact 1.4: Allen Klein was a writers' representative.
  Fact 1.5: Allen Klein was a filmmaker.
  Fact 1.6: Allen Klein was a record label executive.
  Fact 1.7: Allen Klein was most noted for his tough persona.
  Fact 1.8: Allen Klein was most noted for aggressive, innovative negotiation tactics.
  Fact 1.9: Many of Allen Klein's negotiation tactics established higher industry standards for compensating recording artists.
Sentence 2:  He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming one of the most powerful individuals in the music industry during his era.
  Fact 2.0: Allen Klein first scored massive monetary windfalls for Buddy Knox.
  Fact 2.1: Allen Klein first scored massive contractual windfalls for Buddy Knox.
  Fact 2.2: Allen Klein first scored massive monetary windfalls for Jimmy Bowen.
  Fact 2.3: Allen Klein first scored massive contractual windfalls for Jimmy Bowen.
  Fact 2.4: Buddy Knox and Jimmy Bowen were one-hit rockabillies of the late 1950s.
  Fact 2.5: Allen Klein later parlayed his early successes into a position managing Sam Cooke.
  Fact 2.6: Allen Klein eventually managed the Beatles.
  Fact 2.7: Allen Klein eventually managed the Rolling Stones.
  Fact 2.8: Allen Klein managed the Beatles and the Rolling Stones simultaneously.
  Fact 2.9: Allen Klein managed many other artists.
  Fact 2.10: Allen Klein became one of the most powerful individuals in the music industry during his era.

Retrieved items:
<item 1>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 1>

<item 2>
Howie Klein
Howie Klein (born 1950) is an American writer, concert promoter, disc jockey, music producer, record label founder, record label executive, progressive
political activist, and adjunct professor of music.

 He is perhaps best known for his role as President of Reprise Records from 1989 to 2001.

 He appears occasionally as himself in music-related film documentaries and has received accolades for his stance against censorship and for his advocacy of
free speech protection.
</item 2>

<item 3>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 3>

<item 4>
More Hot Rocks (Big Hits &amp; Fazed Cookies)
More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who
usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1972 as the follow-up to the hugely successful "Hot Rocks 1964–1971", it was another success for Klein and ABKCO.
</item 4>

<item 5>
Hot Rocks 1964–1971
Hot Rocks 1964–1971 is the first compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who gained control of the
band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1971, it proved to be The Rolling Stones' biggest-selling release of their career and an enduring and popular retrospective.
</item 5>

<item 6>
George Thorogood
George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.

 His "high-energy boogie-blues" sound became a staple of 1980s rock radio, with hits like his original songs "Bad to the Bone" and "I Drink Alone".

 He has also helped popularize older songs by American icons, such as "Move It on Over", "Who Do You Love?

", and "House Rent Boogie/One Bourbon, One Scotch, One Beer", which became staples of classic rock radio.
</item 6>

<item 7>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 7>

<item 8>
Rock Star Supernova
Rock Star Supernova was a reality television-formed supergroup consisting of drummer Tommy Lee (Mötley Crüe), bassist Jason Newsted (ex-Metallica), guitarist
Gilby Clarke (ex-Guns N' Roses) and singer Lukas Rossi.

 The band was formed during the second season of the Rock Star Reality TV series which was called .
</item 8>

<item 9>
Rock Star (TV series)
Rock Star is a television series produced by Mark Burnett, David Goffin, and Lisa Hennessy in which aspiring singers from all around the world competed to
become the lead singer of a featured group.

 It debuted on CBS on July 11, 2005, to mediocre ratings.

 The show was hosted by television personality and commercial spokeswoman Brooke Burke and Jane's Addiction & Red Hot Chili Peppers lead guitarist Dave Navarro.

 In Australian band INXS chose J.D. Fortune as their new lead singer.

 For , the band Rock Star Supernova chose Lukas Rossi as the lead singer of their new supergroup.
</item 9>

<item 10>
Mick (DJ)
Mick Batyske (known by his stage name Mick, sometimes styled as MICK, and formerly Mick Boogie) is an American DJ and entrepreneur.

 He is an A-list DJ and spun private parties for celebrities including Kanye West, LeBron James, Jay-Z and Will Smith.

 In addition to his mix tape releases, he has performed in venues internationally, including New York City, Dubai, Tokyo, Las Vegas, and Los Angeles.

 As an entrepreneur, he has invested in various start-up companies including Localeur, in which he is also an advisor and consultant.
</item 10>
=== END ITEM id=20240dca1da39aa8 ===

=== ITEM id=72559f23e97cd2db ===
Gold sentences and facts:
Sentence 0: Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
  Fact 0.0: Marc Rosset was born on 7 November 1970.
  Fact 0.1: Marc Rosset is a former professional tennis player.
  Fact 0.2: Marc Rosset is from Switzerland.
  Fact 0.3: Marc Rosset is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
Sentence 1: Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional tennis player.
  Fact 1.0: Maksim Mirnyi, also known as Max Mirnyi, was born on 6 July 1977 in Minsk.
  Fact 1.1: Maksim Mirnyi's Belarusian name is Максім Мікалаевіч Мірны.
  Fact 1.2: Maksim Mirnyi's Russian name is Максим Николаевич Мирный.
  Fact 1.3: Maksim Mirnyi is a Belarusian professional tennis player.

Retrieved items:
<item 1>
Max Mirnyi
Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional
tennis player.
</item 1>

<item 2>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 2>

<item 3>
2001 Milan Indoor – Singles
Marc Rosset was the defending champion at the Milan Indoor tennis tournament but lost in the second round to Greg Rusedski.
</item 3>

<item 4>
Rossana de los Ríos
Rossana de los Ríos (born 16 September 1975) is a retired professional tennis player from Paraguay.

 She was born in the Paraguayan capital, Asunción, and now lives in Miami, USA, with her daughter Ana Paula Neffa de los Ríos and her husband Gustavo Neffa, a
retired Paraguayan Association football player who played for Boca Juniors.

 Rossana achieved a career high singles ranking of No. 51 on 12 September 2001, and No. 52 in doubles on the Sony Ericsson WTA Tour.

 De los Ríos played doubles with Grand Slam champions such as Maria Sharapova, Svetlana Kuznetsova, Arantxa Sánchez Vicario, as well as Jelena Janković and
Dinara Safina.

 Rossana is also a former world number one Junior player, having won the 1992 Roland Garros Juniors event.

 De los Ríos participated in six editions (1994, 2001, 2002, 2003, 2009 and 2010) of the Australian Open.
</item 4>

<item 5>
2010 Pilot Pen Tennis – Men's Doubles
Julian Knowle and Jürgen Melzer were the defending champions, but Melzer chose not to participate this year.As a result, Knowle partnered with Andy Ram, but
they lost to Robert Lindstedt and Horia Tecău in the semifinals.

 Lindstedt and Tecău went on to win the tournament, after defeating Rohan Bopanna and Aisam-ul-Haq Qureshi in the final 6–4, 7–5.
</item 5>

<item 6>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 6>

<item 7>
2005 BNP Paribas Masters – Doubles
Jonas Björkman and Todd Woodbridge were the defending champions, but did not participate together this year.

 Björkman partnered Max Mirnyi, losing in the first round.

 Woodbridge retired from professional tennis earlier in the year.
</item 7>

<item 8>
2004 Canada Masters – Doubles
The 2004 Canada Masters – Doubles was the men's singles event of the one hundred and fifteenth edition of the Canada Masters; a WTA Tier I tournament and the
most prestigious men's tennis tournament held in Canada.

 Mahesh Bhupathi and Max Mirnyi were the defending champions.

 They were both present but did not compete together.

 Mirnyi partnered with Jonas Björkman, but Bhupathi and partner Leander Paes defeated them 6–4, 6–2, in the final.
</item 8>

<item 9>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 9>

<item 10>
2010 Australian Open – Mixed Doubles
Sania Mirza and Mahesh Bhupathi were the defending champions.

 They accepted the invitation for the tournament, but withdrew before their match against Carly Gullickson and Bernard Tomic and were replaced by Akgul
Amanmuradova and Rik de Voest.
</item 10>
=== END ITEM id=72559f23e97cd2db ===
