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

=== ITEM id=35fc844f12b61366 ===
Gold sentences and facts:
Sentence 0: Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.
  Fact 0.0: Eucryphia is a small genus.
  Fact 0.1: Eucryphia consists of trees and large shrubs.
  Fact 0.2: Eucryphia is native to the south temperate regions of South America.
  Fact 0.3: Eucryphia is native to coastal eastern Australia.
Sentence 1:  Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.
  Fact 1.0: Eucryphia is sometimes placed in a family of its own, the Eucryphiaceae.
  Fact 1.1: More recent classifications place Eucryphia in the Cunoniaceae.
Sentence 2:  There are seven species, two in South America and five in Australia, and several named hybrids.
  Fact 2.0: Eucryphia has seven species.
  Fact 2.1: Two Eucryphia species are in South America.
  Fact 2.2: Five Eucryphia species are in Australia.
  Fact 2.3: Eucryphia has several named hybrids.
Sentence 3: Lepidozamia is a genus of two species of cycad, both endemic to Australia.
  Fact 3.0: Lepidozamia is a genus of cycad.
  Fact 3.1: Lepidozamia has two species.
  Fact 3.2: Both Lepidozamia species are endemic to Australia.

Retrieved items:
<item 1>
Cassiope
Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.

 They are native to the Arctic and north temperate montane regions.

 The genus is named after Cassiopeia of Greek mythology.

 Common names, shared with several other similar related genera, include heather and heath.

 They have scale-like leaves lying against the stems, and produce solitary bell-shaped flowers in late spring.

 Though hardy, flowers can be damaged by late frosts.
</item 1>

<item 2>
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
</item 2>

<item 3>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 3>

<item 4>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 4>

<item 5>
Eucommia
Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.

 The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in
herbology such as traditional Chinese medicine.
</item 5>

<item 6>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 6>

<item 7>
Ammocharis
Ammocharis is a small genus from sub-Saharan Africa, in the Amaryllidaceae family (subfamily Amaryllidoideae) which includes 7 species distributed in Africa.

 The plant grows as a succulent, above-ground bulb, preferring seasonally wet, hot, sandy soils and full sun.
</item 7>

<item 8>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 8>

<item 9>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 9>

<item 10>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 10>
=== END ITEM id=35fc844f12b61366 ===

=== ITEM id=7e33d906e30f5adf ===
Gold sentences and facts:
Sentence 0: Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
  Fact 0.0: Mauritz Stiller was born on 17 July 1883.
  Fact 0.1: Mauritz Stiller died on 18 November 1928.
  Fact 0.2: Mauritz Stiller was a Finnish-Swedish film director.
  Fact 0.3: Mauritz Stiller was best known for discovering Greta Garbo.
  Fact 0.4: Mauritz Stiller brought Greta Garbo to America.
Sentence 1: Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.
  Fact 1.0: Johannes Grenzfurthner was born in 1975 in Vienna.
  Fact 1.1: Johannes Grenzfurthner is an award-winning Austrian artist.
  Fact 1.2: Johannes Grenzfurthner is an Austrian filmmaker.
  Fact 1.3: Johannes Grenzfurthner is an Austrian writer.
  Fact 1.4: Johannes Grenzfurthner is an Austrian actor.
  Fact 1.5: Johannes Grenzfurthner is an Austrian curator.
  Fact 1.6: Johannes Grenzfurthner is an Austrian theatre director.
  Fact 1.7: Johannes Grenzfurthner is an Austrian lecturer.

Retrieved items:
<item 1>
Austrian director-screenwriter Stefan Ruzowitzky made *The Inheritors* (1998), Austria’s unsuccessful Oscar submission, and *The Counterfeiters* (2007), a WWII
drama about Nazi forgery at Sachsenhausen that won the foreign-language-film Oscar; Devid Striesow played Bernhard Krüger. German director Wolfgang Becker is
known for *Good Bye, Lenin!* (2003), starring Daniel Brühl and Katrin Saß. Other entries
</item 1>

<item 2>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 2>

<item 3>
John Steiner
John Steiner (born 7 January 1941 in Chester, Cheshire) is an English actor.

 Tall, thin and gaunt, Steiner attended the Royal Academy of Dramatic Arts and worked for a few years at the BBC.

 Steiner featured in a lead role in a television production of "Design for Living" by Noël Coward.

 Later he found further work primarily in films including "Marat/Sade" (1967), and the original "Bedazzled" (1967) with Peter Cook and Dudley Moore.

 In 1969, Steiner was hired to play a part in the spaghetti western "Tepepa", and also appeared opposite Franco Nero in "White Fang", directed by Lucio Fulci.
</item 3>

<item 4>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 4>

<item 5>
Johannes Grenzfurthner
Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.

 He is known as the founder, conceptualist and artistic director of monochrom, an international art and theory group.

 Most of his artworks are labelled "monochrom".
</item 5>

<item 6>
Mauritz Stiller
Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
</item 6>

<item 7>
Traceroute (film)
Traceroute is a 2016 Austrian/American documentary film directed by Johannes Grenzfurthner.

 The autobiographical documentary and road movie deals with the history, politics and impact of nerd culture.

 Grenzfurthner calls his film a "personal journey into the uncharted depths of nerd culture, a realm full of dangers, creatures and more or less precarious
working conditions", an attempt to "chase the ghosts of nerddom's past, present and future."

 The film was co-produced by art group monochrom and Reisenbauer Film.

 It features music by Kasson Crooker, Hans Nieswandt, and many others.
</item 7>

<item 8>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 8>

<item 9>
Willi Forst
Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.

 As a debonair actor he was a darling of the German-speaking film audiences, as a director, one of the most significant makers of the Viennese period musical
melodramas and comedies of the 1930s known as "Wiener Filme".

 From the mid-1930s he also recorded many records, largely of sentimental Viennese songs, for the Odeon Records label owned by Carl Lindström AG.
</item 9>

<item 10>
The entries survey European films and music drama, 1913–1961. Swedish director Mauritz Stiller’s works include *The Blizzard* (a student saves his family
estate), *Erotikon* (a professor’s wife attracts suitors), and *The Saga of Gösta Berling*, Greta Garbo’s breakout film. Hungarian director Béla Gaál made
comedies including *The Dream Car*. Other highlights: *Triumph of the Will*, Riefenstahl’s
</item 10>
=== END ITEM id=7e33d906e30f5adf ===

=== ITEM id=55c8ec0a49539d8a ===
Gold sentences and facts:
Sentence 0:  The name comes from the City of Starachowice, where the factory is located.
  Fact 0.0: The name comes from the City of Starachowice.
  Fact 0.1: The factory is located in the City of Starachowice.
Sentence 1:  It was sold in various countries for many years (not only in the Eastern Bloc; for example it was used by the Yemen Army).
  Fact 1.0: The item was sold in various countries for many years.
  Fact 1.1: The item was sold in countries beyond the Eastern Bloc.
  Fact 1.2: The item was used by the Yemen Army.
Sentence 2: The Eastern Bloc was the group of socialist states of Central and Eastern Europe, generally the Soviet Union and the countries of the Warsaw Pact.
  Fact 2.0: The Eastern Bloc was the group of socialist states of Central and Eastern Europe.
  Fact 2.1: The Eastern Bloc generally included the Soviet Union.
  Fact 2.2: The Eastern Bloc generally included the countries of the Warsaw Pact.

Retrieved items:
<item 1>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 1>

<item 2>
Monochrom’s fictional Soviet Unterzögersdorf, inspired by creator Johannes Grenzfurthner’s family stories, imagines the USSR’s last enclave in Austria. Liberal
arts were skills for civic participation. The Eastern Bloc comprised Soviet-aligned socialist states; the Warsaw Pact was its military alliance. After WWII,
about 31 million people, including Germans, were displaced across Central and Eastern Europe. The Curzon Line informed Poland’s eastern-border debate; Kresy,
incorporated into Soviet republics
</item 2>

<item 3>
Leader Trucks
Leader found their niche in the heavy transport market and built as a rigid 4x4 and 6x6 configuration for tray, tipper and agitator applications, they offered
reliable machines in the industries of earth-moving and off-road construction.

 Leader was notable for being the first manufacturer to fit Caterpillar engines to trucks, to fit automatic transmissions to diesel trucks in assembly, and the
first truck manufacturer in Australia to offer disc brakes.

 Leader boasted its components to be 80 percent Australian built with only the engine, transmission and steering box imported.

 By 1980, Leader had sold its 1000th truck in Australia and had also sold trucks to New Zealand, South Africa, Hong Kong, China and Indonesia.
</item 3>

<item 4>
Shaanxi Automobile Group
Shaanxi Automobile Group Co., Ltd. is a Chinese bus and truck manufacturer headquartered in Xi'an, Shaanxi.

 It employs around 23,000 employees.

 It manufactures heavy duty truck, buses (chassis), medium-size heavy truck and heavy duty axles for trucks.

 It utilizes Magna Steyr and MAN SE technologies.

 Its bus chassis are sold under the Eurostar Bus brand.
</item 4>

<item 5>
Daimler, once the world’s largest truckmaker, sold 3 million vehicles in 2016. Mouawad, founded in Beirut in 1891, is a Geneva/Dubai jeweler; Dior is controlled
by Bernard Arnault and holds a major LVMH stake. Microsoft My Phone backed up Windows Mobile data (2009–11). Samsung products ranged from budget phones to
Galaxy flagships and tablets. SAIC-GM-Wuling sold 1.445 million vehicles
</item 5>

<item 6>
J. A. Baczewski
J. A. Baczewski was a Polish company most well known for its fine spirits such as vodka and gin.

 The factory, dating back to the late 18th century, was based in Lwów (Lviv) and until 1939 was one of two most popular Polish export goods.
</item 6>

<item 7>
FSC Star
Fabryka Samochodów Ciężarowych "Star" was a Polish truck manufacturer.

 The name comes from the City of Starachowice, where the factory is located.

 Their first vehicle was the "Star 20" in 1948.

 The most popular product was the "Star 266".

 The 266 model offered very good quality and powerful engines for a low price.

 It was sold in various countries for many years (not only in the Eastern Bloc; for example it was used by the Yemen Army).

 For many years FSC Star was a state-owned company.

 Star is now owned by MAN AG who eliminated the brand in January 2009.
</item 7>

<item 8>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 8>

<item 9>
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 9>

<item 10>
Daimler AG
Daimler AG (] ) is a German multinational automotive corporation.

 Daimler AG is headquartered in Stuttgart, Baden-Württemberg, Germany.

 As of 2014, Daimler owns or has shares in a number of car, bus, truck and motorcycle brands including Mercedes-Benz, Mercedes-AMG, Smart Automobile, Detroit
Diesel, Freightliner, Western Star, Thomas Built Buses, Setra, BharatBenz, Mitsubishi Fuso, MV Agusta as well as shares in Denza, KAMAZ, Beijing Automotive
Group.

 The luxury Maybach brand was terminated at the end of 2012, but revived in April 2015 as "Mercedes-Maybach" versions of the Mercedes-Benz S-Class and G-Class.

 In 2016 Daimler sold 3.0 million vehicles.

 By unit sales, Daimler is the thirteenth-largest car manufacturer and is the largest truck manufacturer in the world.

 In addition to automobiles, Daimler manufactures buses and provides financial services through its Daimler Financial Services arm.

 The company is a component of the Euro Stoxx 50 stock market index.
</item 10>
=== END ITEM id=55c8ec0a49539d8a ===

=== ITEM id=37812ad806b01b7b ===
Gold sentences and facts:
Sentence 0: "Grow Some Funk of Your Own" is a song by Elton John.
  Fact 0.0: 'Grow Some Funk of Your Own' is a song by Elton John.
Sentence 1: Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.
  Fact 1.0: Sir Elton Hercules John was born Reginald Kenneth Dwight.
  Fact 1.1: Sir Elton Hercules John was born on 25 March 1947.
  Fact 1.2: Sir Elton Hercules John is an English singer.
  Fact 1.3: Sir Elton Hercules John is an English pianist.
  Fact 1.4: Sir Elton Hercules John is an English composer.
Sentence 2:  John owned Watford Football Club from 1976 to 1987, and 1997 to 2002.
  Fact 2.0: Elton John owned Watford Football Club from 1976 to 1987.
  Fact 2.1: Elton John owned Watford Football Club from 1997 to 2002.

Retrieved items:
<item 1>
Elton John
Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.

 He has worked with lyricist Bernie Taupin as his songwriting partner since 1967; they have collaborated on more than 30 albums to date.

 In his five-decade career Elton John has sold more than 300million records, making him one of the best-selling music artists in the world.

 He has more than fifty Top 40 hits, including seven consecutive No. 1 US albums, 58 "Billboard" Top 40 singles, 27 Top 10, four No. 2 and nine No. 1.

 For 31 consecutive years (1970–2000) he had at least one song in the "Billboard" Hot 100.

 His tribute single, re-penned in dedication to the late Princess Diana, "Candle in the Wind 1997" sold over 33million copies worldwide and is the best-selling
single in the history of the UK and US singles charts.

 He has also composed music, produced records, and has occasionally acted in films.
</item 1>

<item 2>
Firm Biz
"Firm Biz" is the first single by the Hip hop supergroup The Firm from their singular collaborative LP "The Album".

 The song was produced by L.E.S, who based the song's track on a sample of Teena Marie's 1981 hit "Square Biz".

 "Firm Biz" also features a chorus performed by former En Vogue singer Dawn Robinson, reworked from the "Square Biz" chorus.

 AZ raps the first verse of the song, Nas the second, and Foxy Brown the third and final verse.

 The song never reached any US chart positions, but was a UK Top 20 hit in 1997.

 A remix of the song was released, featuring a verse from Half-A-Mill and a chorus by Mary J. Blige.

 The word "feds" is censored in the explicit version, in verse 2, performed by Nas, when he says "Never that though Black .4-4's for feds".
</item 2>

<item 3>
The Happenings
The Happenings are a pop music group that originated in the 1960s.

 Members of the original group, created in the spring of 1961 and initially called "The Four Grads" because all had just graduated from high school in Paterson,
New Jersey, were Bob Miranda, David Libert, Tom Giuliano, and Ralph DiVito.

 In 1968 DiVito was replaced by Bernie LaPorta and Lenny Conforti also joined to play drums in the touring band.

 Both LaPorta and Conforti took a hiatus from the northern New Jersey band, The Emerald Experience, to play and tour with The Happenings.

 That lineup performed mostly at colleges and universities until 1970, when Libert left the band to manage other groups, including George Clinton and
Parliament-Funkadelic, Living Colour, Brian Auger, Vanilla Fudge, The Runaways (Cherie Currie, Joan Jett, Lita Ford), Mother's Finest, Alice Cooper and Evelyn
"Champagne" King.
</item 3>

<item 4>
Beasts of Bourbon
The Cramps-influenced swamp-rock of old had been discarded for a more adventurous slab of gutbucket blues and avant-garde weirdness.

 Perkins' voice had matured into an authentic blues growl".

 Their fifth studio album, "Gone" (January 1997), reached the Top 50 ARIA Albums Chart.

 Their seventh studio album, ""Little Animals" (21 April 2007) on Albert Productions, which also peaked into the Top 50.
</item 4>

<item 5>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 5>

<item 6>
Grow Some Funk of Your Own
"Grow Some Funk of Your Own" is a song by Elton John.

 It was released as a single in 1976 and comes from his album "Rock of the Westies".

 It shared its A-side status with "I Feel Like a Bullet (In the Gun of Robert Ford)".

 The song is considered to be one of John's heavier rock songs.

 The song went to #14 on the "Billboard" Hot 100, but in Britain broke a five-year run of successful singles by failing to reach the Top 50 despite extensive
radio play.

 Although the singles chart listed only "Funk" and not "Bullet", the latter song was given equal billing in later "Billboard" singles chart books.
</item 6>

<item 7>
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
</item 7>

<item 8>
Elton John
He has also composed music, produced records, and has occasionally acted in films.

 John owned Watford Football Club from 1976 to 1987, and 1997 to 2002.

 He is an honorary Life President of the club, and in 2014 had a stand named after him at the club's home stadium.
</item 8>

<item 9>
Blitz Kids
The Blitz Kids were a group of young people who frequented the weekly Blitz club-night in Covent Garden, London in 1979-80, and are credited with launching the
New Romantic subcultural movement.

 It was Andrew Czezowski (Ex manager of The Damned) and Susan Carrington who went on to start The Roxy club with Barry Jones that introduced Steve Strange and
Rusty Egan to the Blitz in Covent Garden, which started the New Romantic Movement.

 Among the many attendees were Rusty Egan, Steve Strange, Boy George, Marilyn, Alice Temple, Perri Lister, Princess Julia, Philip Sallon, Carl Teper and Martin
Degville (later to be the frontman of Tony James' Sigue Sigue Sputnik).

 Crucially, the club lay between two art colleges (St Martin's School and Central School) and became a testbed for student fashion designers who set London
ablaze during the 1980s.
</item 9>

<item 10>
Knitting Factory Records
Knitting Factory Records is an independent American music label that is notable for promoting a variety of artists, including the music of deceased Nigerian
political activist Fela Kuti.

 The label promotes a variety of music artists including Ages and Ages, Ash Black Bufflo, Cuong Vu, Graham Haynes, Femi Kuti, Gary Lucas, Lumerians, Thomas
Chapin, Patrolled By Radar, Joe Morris, Rachid Taha, Seun Kuti, and Shilpa Ray and her Happy Hookers.
</item 10>
=== END ITEM id=37812ad806b01b7b ===

=== ITEM id=619709ccebed4d6a ===
Gold sentences and facts:
Sentence 0: Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.
  Fact 0.0: Henderson Point is a census-designated place (CDP).
  Fact 0.1: Henderson Point is in Harrison County, Mississippi, United States.
Sentence 1:  Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
  Fact 1.0: Gulfport is a city adjoining Biloxi.
  Fact 1.1: Biloxi is a county seat of Harrison County.

Retrieved items:
<item 1>
Henderson, Texas
Henderson is a city in Rusk County, northeast Texas, United States.

 The population was 13,712 at the 2010 census.

 It is the county seat of Rusk County.

 Henderson is named for James Pinckney Henderson, the first governor of Texas.
</item 1>

<item 2>
Henderson Point, Mississippi
Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.

 It is part of the Gulfport–Biloxi Metropolitan Statistical Area.

 The population was 170 at the 2010 census.
</item 2>

<item 3>
Jasper County Community Unit School District 1
Jasper County Community Unit School District 1 is a unified school district based in Jasper County's county seat of Newton, Illinois; it is the only school
district in the county and is, consequently, the main educational body in all of Jasper County, although it serves portions of Effingham County and Cumberland
County as well.

 This school district is composed of six schools in total; four elementary schools, one junior high school, and one high school.

 There is also a prekindergarten program run at the high school of the district should parents wish to enroll their children early.

 Willow Hill Elementary School, which is located in the village of its namesake, serves only kindergarteners; its proximity to the county seat and central
position in the county allows its students to dawn from all parts of the county and still have easy access to the elementary school they will attend.

 The principal of Willow Hill is Dave Parker.

 Grove Elementary School is located in Island Grove, Illinois, the highest point of elevation in the county.

 Grove Elementary School educates students from kindergarten to grade six, and it runs a prekindergarten program as well.

 Craig Carr is the principal of this school.
</item 3>

<item 4>
Henderson County, Kentucky
Henderson County is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 46,250.

 The county seat is Henderson.

 The county was formed in 1798 and named for Colonel Richard Henderson who purchased 17,000,000 acre of land from the Cherokee Indians, part of which would
eventually make up the county.
</item 4>

<item 5>
Paulding, Mississippi
Paulding is an unincorporated community in and one of the two county seats of Jasper County, Mississippi.

 It is the only unincorporated county seat in Mississippi.

 Settled in 1833, it was named by United States settlers in honor of Revolutionary War hero John Paulding.

 After its citizens refused to contribute to a new railroad, the community was bypassed in favor of Bay Springs, Mississippi, which was designated a railroad
stop to the west and the second county seat.

 It attracted more development and industry.
</item 5>

<item 6>
Raeford, North Carolina
Raeford is a city in Hoke County, North Carolina, United States.

 The population was 4,626 at the 2010 census.

 It is the county seat of Hoke County.

 The county was named after Confederate General Robert F. Hoke, a North Carolina native.
</item 6>

<item 7>
Hendersonville, North Carolina
Hendersonville is a city in Henderson County, North Carolina United States.

 This city is 22 miles south of Asheville.

 It is the county seat of Henderson County.

 Like the county, the city is named for 19th-century North Carolina Supreme Court Chief Justice Leonard Henderson.
</item 7>

<item 8>
McIntosh County Seat War
The McIntosh County Seat War was a dispute in Oklahoma over the location of the McIntosh County seat that took place between 1907 and 1909.

 Following a pair of elections that resulted in the town of Checotah being designated as the new county seat, the people of Eufaula refused to hand over the
county records.

 As a result, a group of heavily armed men from Chectotah attempted to seize the records, but were forced to surrender during the gunbattle that ensued.

 One year later, after another close election, Eufaula became the permanent county seat.
</item 8>

<item 9>
Piney Woods Country Life School
The Piney Woods Country Life School (or The Piney Woods School) is a co-educational independent historically African-American boarding school for grades 9-12 in
Piney Woods, unincorporated Rankin County, Mississippi.

 It is 21 mi south of Jackson.

 It is one of four remaining historically African-American boarding schools in the United States.

 It is currently the largest African-American boarding school, as well as being the second oldest continually operating African-American boarding school.
</item 9>

<item 10>
Biloxi, Mississippi
Biloxi ( ) is a city in Harrison County, Mississippi, United States.

 The 2010 United States Census recorded the population as 44,054, and in 2016 the estimated population was 45,975.

 Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
</item 10>
=== END ITEM id=619709ccebed4d6a ===

=== ITEM id=46d45a7f93ba3948 ===
Gold sentences and facts:
Sentence 0:  Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer Jimmy Chamberlin.
  Fact 0.0: The band was formed by Corgan and guitarist James Iha.
  Fact 0.1: The band was formed in Chicago, Illinois, in 1988.
  Fact 0.2: The band quickly gained steam with the addition of bassist D'arcy Wretzky.
  Fact 0.3: The band quickly gained steam with the addition of drummer Jimmy Chamberlin.
Sentence 1: Greek Fire is an American rock band from St. Louis, Missouri.
  Fact 1.0: Greek Fire is an American rock band.
  Fact 1.1: Greek Fire is from St. Louis, Missouri.

Retrieved items:
<item 1>
Greek Fire (band)
Greek Fire is an American rock band from St. Louis, Missouri.

 The band was formed in 2008 by members of Story of the Year and Maybe Today.

 Since formation, Greek Fire has released a self-titled EP, a single titled "Doesn't Matter Anyway", on August 16, 2011, they released their debut, full-length
album, "Deus Ex Machina", and have recently announced a new addition to the "Lost/Found" EPs titled "Broken" set to be released before "Found".
</item 1>

<item 2>
Zeitgeist (The Smashing Pumpkins album)
Zeitgeist is the seventh album by American alternative rock band The Smashing Pumpkins, released on July 10, 2007 in the United States and Canada.

 It was the first album The Smashing Pumpkins released after their 2000 disbandment and 2005 reunion.

 The album was produced by Roy Thomas Baker, Billy Corgan, Jimmy Chamberlin, and Terry Date.

 The album would be Chamberlin's last with the band before his departure in 2009.

 The album debuted strongly, but sales soon decreased, and critical reception was mixed.

 It was certified Gold in the United States on February 1, 2008.
</item 2>

<item 3>
1979 (song)
"1979" is a song by American alternative rock band the Smashing Pumpkins.

 Released in 1996 as the second single from their third studio album, "Mellon Collie and the Infinite Sadness", "1979" was written by frontman Billy Corgan, and
features loops and samples that were uncharacteristic of previous Smashing Pumpkins songs.

 The song was written as a coming of age story by Corgan.

 In the year 1979, Corgan was 12 and this is what he considered his transition into adolescence.

 The song was popular with critics and fans; Allmusic's Amy Hanson called it a "somewhat surprising hit".

 The song was nominated for the Record of the Year and Best Rock Performance by a Duo or Group with Vocal at the 39th Annual Grammy Awards, and won an MTV Video
Music Award for Best Alternative Video.

 In 2012, it was voted the second-best Smashing Pumpkins song by Rolling Stone magazine readers.
</item 3>

<item 4>
List of the Smashing Pumpkins band members
The Smashing Pumpkins are an alternative rock band formed in Chicago, Illinois in 1988.

 The band was formed by guitarist/vocalist Billy Corgan and guitarist James Iha after the demise of Corgan's first band, The Marked.

 Since its inception, The Smashing Pumpkins has gone through multiple line-up changes, with Corgan the only consistent member.
</item 4>

<item 5>
The Aeroplane Flies High
The Aeroplane Flies High is a five-disc box set released by American alternative rock band The Smashing Pumpkins in 1996.

 It contains expanded versions of the five singles from their album "Mellon Collie and the Infinite Sadness" and also included a 44-page booklet with pictures
and writings by the band's lead singer Billy Corgan, as well as lyrics.

 A limited edition release, the box reached number 42 on the "Billboard" charts, and sold 300,000 units (1.5 million discs in all), generating a platinum disc
for the band.

 Originally intended to be limited to 200,000 copies, Virgin Records produced more after the original run sold out due to overwhelming and unexpected demand.

 The album was remastered in 2013 under the supervision of frontman Billy Corgan and reissued on vinyl and as a CD/DVD box set.
</item 5>

<item 6>
Mike Byrne (musician)
Michael William Byrne (born February 6, 1990) is a drummer who was a member of the band The Smashing Pumpkins.

 When Pumpkins frontman Billy Corgan called for rehearsals to replace Jimmy Chamberlin, Byrne auditioned and was chosen out of thousands of applicants.

 He played drums in Sky Saxon tribute band Spirits in the Sky for 6 shows in August 2009, along with Corgan, Kerry Brown, Kevin Dippold, Mark Tulin, Linda
Strawberry, Ysanne Spevack, Mark Weitz, and Dave Navarro, and also performs drums on the ongoing Pumpkins 44-song project, "Teargarden by Kaleidyscope".

 As a part of "Teargarden", Mike contributed drumming duties on "Oceania", the Pumpkins' eighth full-length album.

 On April 17, 2010, Byrne played his first show under the Smashing Pumpkins moniker in celebration of Record Store Day.
</item 6>

<item 7>
Muzzle (song)
"Muzzle" is a song by American alternative rock band The Smashing Pumpkins from their third album, "Mellon Collie and the Infinite Sadness".

 It was one of the last songs written by Billy Corgan for "Mellon Collie", with the song's lyrics referring to what Corgan thought the public's perception was
of him at the time.

 It was rumored to be the Smashing Pumpkins fifth and final single from this album, as is evidenced by the fact that a promotional single for the song was
issued to radio stations worldwide.

 However, the song "Thirty-Three" was released as the fifth and final single instead.
</item 7>

<item 8>
Jimmy Chamberlin
James Joseph Chamberlin (born June 10, 1964) is an American drummer and record producer.

 He is best known as the drummer for the alternative rock band The Smashing Pumpkins.

 Following the 2000 breakup of the band, Chamberlin joined Pumpkins frontman Billy Corgan in the supergroup Zwan and also formed his own group, the Jimmy
Chamberlin Complex.

 In 2005, Chamberlin joined Corgan in reforming The Smashing Pumpkins; he eventually left the group in March 2009, though he returned again in 2015 for a summer
tour.

 He performed in the group Skysaw until 2012.

 He is currently active under the Jimmy Chamberlin Complex name.

 In addition to his current work as CEO, Chamberlin has joined Chicago jazz saxophonist Frank Catalano for a string of 2013–15 performances in the Chicago area.

 An EP by Catalano and Chamberlin "Love Supreme Collective - EP" was released on the 29th of July 2014.
</item 8>

<item 9>
Pinhead Gunpowder
Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.

 The band currently consists of Aaron Cometbus (drums, lyrics), Bill Schneider (bass), Billie Joe Armstrong (guitar, vocals) and Jason White (guitar, vocals).

 The band's name comes from a brand of "high octane" green tea served at the Arcata co-op and discovered by Aaron Cometbus during one of his many dumpster
diving adventures.
</item 9>

<item 10>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 10>
=== END ITEM id=46d45a7f93ba3948 ===

=== ITEM id=6ee1188a4d8980ad ===
Gold sentences and facts:
Sentence 0:  His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files" (1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
  Fact 0.0: The subject of this sentence portrayed Bret Maverick in the series Maverick from 1957 to 1962.
  Fact 0.1: The subject of this sentence portrayed Bret Maverick in the series Bret Maverick from 1981 to 1982.
  Fact 0.2: The subject of this sentence portrayed Jim Rockford in the series The Rockford Files from 1974 to 1980.
  Fact 0.3: The subject of this sentence portrayed Jim Egan in the series 8 Simple Rules from 2003 to 2005.
Sentence 1: The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.
  Fact 1.0: The Notebook is a 2004 film.
  Fact 1.1: The Notebook is an American romantic drama film.
  Fact 1.2: The Notebook was directed by Nick Cassavetes.
  Fact 1.3: The Notebook is based on a 1996 novel.
  Fact 1.4: The 1996 novel has the same name as the film The Notebook.
  Fact 1.5: The 1996 novel was written by Nicholas Sparks.

Retrieved items:
<item 1>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 1>

<item 2>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 2>

<item 3>
Mister Buddwing
Mister Buddwing is a 1966 American film drama directed by Delbert Mann and starring James Garner.
</item 3>

<item 4>
The Notebook
The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.

 The film stars Ryan Gosling and Rachel McAdams as a young couple who fall in love in the 1940s.

 Their story is narrated from the present day by an elderly man (portrayed by James Garner) telling the tale to a fellow nursing home resident (played by Gena
Rowlands, who is Cassavetes's mother).
</item 4>

<item 5>
Cash McCall
Cash McCall is a 1960 American romantic drama film in Technicolor from Warner Bros., produced by Henry Blanke, directed by Joseph Pevney, that stars James
Garner and Natalie Wood.

 It is based upon the novel of the same name by Cameron Hawley about a man who buys moribund businesses in order to refurbish them and then sell them on at
considerable profit.

 The film's screenplay is by Lenore J. Coffee and Marion Hargrove.
</item 5>

<item 6>
James Garner filmography
The following is the filmography for American actor James Garner.

 Garner was known for prominent roles in films such as "Sayonara" (1957) with Marlon Brando, "Cash McCall" (1960) with Natalie Wood, "The Great Escape" (1963)
with Steve McQueen, "The Thrill of It All" (1963) with Doris Day, "Move Over, Darling" (1963) with Doris Day, "The Wheeler Dealers" (1963) with Lee Remick,
Paddy Chayevsky's "The Americanization of Emily" (1964) with Julie Andrews, "Hour of the Gun" (1967), "Marlowe" (1969) with Bruce Lee, "Support Your Local
Sheriff!

"
</item 6>

<item 7>
List of American films of 2004
A list of American films released in 2004. "

Million Dollar Baby" won the 2004 Academy Award for Best Picture and "Crash" won the 2005 Academy Award for Best Picture. "

The Aviator" won the BAFTA Award for Best Film and the Golden Globe Award for Best Motion Picture – Drama. "

Sideways" won the Golden Globe Award for Best Motion Picture – Musical or Comedy and the Satellite Award for Best Film – Musical or Comedy. "

Hotel Rwanda" won the Satellite Award for Best Film – Drama.
</item 7>

<item 8>
Bride and Prejudice
Bride and Prejudice is a 2004 romantic drama film directed by Gurinder Chadha.

 The screenplay by Chadha and Paul Mayeda Berges is a Bollywood-style adaptation of "Pride and Prejudice" by Jane Austen.

 It was filmed primarily in English, with some Hindi and Punjabi dialogue.

 The film released in the United States on 11 February 2005 and was well received by critics.
</item 8>

<item 9>
Milan (2004 film)
Milan is a 2004 romantic drama film released under Star Cinema, ABS-CBN Film Productions, Inc. in the Philippines.

 It stars Claudine Barretto and Piolo Pascual.

 It is a love story written by Raymond Lee, and under Olivia Lamasan's direction.

 The movie made more than P100 million in the box office.
</item 9>

<item 10>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 10>
=== END ITEM id=6ee1188a4d8980ad ===

=== ITEM id=10ff1c8291dd282a ===
Gold sentences and facts:
Sentence 0:  In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.
  Fact 0.0: The Eleanor and John Y. Brown Jr. Foundation gave a grant in 1972.
  Fact 0.1: The 1972 grant from the Eleanor and John Y. Brown Jr. Foundation led to the construction of the current facility.
Sentence 1:  He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.
  Fact 1.0: The 55th governor of Kentucky served from 1979 to 1983.
  Fact 1.1: The 55th governor of Kentucky may be best known for building Kentucky Fried Chicken (KFC) into a multimillion-dollar restaurant chain.

Retrieved items:
<item 1>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
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
Rochelle Buffenstein
Rochelle (Shelley) Buffenstein is a staff scientist at Calico, an Alphabet, Inc. funded research outfit investigating aging.

 Previously, she had been a professor of Physiology at theBarshop Institute for Longevity and Aging Studies at the University of Texas Health Science Center at
San Antonio.

 Her research focuses on comparative vertebrate physiology, energetics, cancer biology, and aging.

 She has worked with marsupials, mole-rats, tenrecs, bats, subterranean mammals, and primates.

 Her best known work involves exceptional aging -- specifically, why naked mole-rats live for so much longer than other rodents.

 Her lab has investigated theories of aging including oxidative damage theory, the advanced glycation end product theory, and the telomere theory.
</item 3>

<item 4>
Panera Bread
Panera Bread Company is an American chain of bakery-café fast casual restaurants in the United States and Canada.

 Its headquarters are in Sunset Hills, Missouri, a suburb of St. Louis, and operates as Saint Louis Bread Company in the St. Louis metropolitan area.

 Offerings include soups, salads, pasta, sandwiches, specialty drinks, and bakery items.
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
VIP's
VIP's, alternatively written Vip's, is a defunct restaurant chain in the Western United States that operated from 1968 until the late 1980s, based in Salem,
Oregon.

 With more than 50 locations, it was once the largest restaurant chain based in Oregon.

 It was a Denny's-style restaurant, a type that was commonly known at that time as a "coffee shop" but is now more commonly known as a casual dining restaurant.

 Most restaurants were located near freeways and were open 24 hours.

 At its peak, the chain had locations in five states: Oregon, Washington, Idaho, Nevada and northern California.
</item 7>

<item 8>
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
</item 8>

<item 9>
McAlister's Deli
McAlister's Deli is an American chain of fast casual restaurants founded in 1989 in Oxford, Mississippi by retired dentist Dr. Don Newcomb.

 There are currently over 400 locations in 26 states, ranging from Virginia in the East to Florida in the South to Arizona in West to Michigan in the North.

 The menu includes deli sandwiches, "Texas-size" spuds (baked potatoes), soups, salads, and desserts, as well as catering items such as sandwich trays and boxed
lunches.

 The chain is also known for its McAlister's Famous Sweet Tea, which is available by the glass or by the gallon.
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
=== END ITEM id=10ff1c8291dd282a ===

=== ITEM id=c94ebc224a822108 ===
Gold sentences and facts:
Sentence 0: Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss is a book written in 2003 by Dr. Joel Fuhrman, which he revised in 2011.
  Fact 0.0: Eat to Live is a book.
  Fact 0.1: The full title of Eat to Live is 'Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss'.
  Fact 0.2: Eat to Live was written in 2003.
  Fact 0.3: Eat to Live was written by Dr. Joel Fuhrman.
  Fact 0.4: Dr. Joel Fuhrman revised Eat to Live in 2011.
Sentence 1: Joel Fuhrman (born December 2, 1953) is an American celebrity doctor who advocates what he calls a micronutrient-rich diet.
  Fact 1.0: Joel Fuhrman was born on December 2, 1953.
  Fact 1.1: Joel Fuhrman is American.
  Fact 1.2: Joel Fuhrman is a celebrity doctor.
  Fact 1.3: Joel Fuhrman advocates a diet that he calls a micronutrient-rich diet.

Retrieved items:
<item 1>
Diet food
Diet food (or dietetic food) refers to any food or beverage whose recipe is altered to reduce fat, carbohydrates, and/or sugar in order to make it part of a
weight loss program or diet.

 Such foods are usually intended to assist in weight loss or a change in body type, although bodybuilding supplements are designed to aid in gaining weight or
muscle.
</item 1>

<item 2>
Judy Mazel
Judy Mazel (December 20, 1943 – October 12, 2007) was the American weight loss advocate and author of the ""The Beverly Hills Diet"", which became a 1981 best
seller and a nationwide diet craze.

 Judy Mazel was a devout practicing Jew and was raised in a conservative Ashkenazi Jewish home.
</item 2>

<item 3>
Management of obesity
Management of obesity can include lifestyle changes, medications, or surgery.

 The main treatment for obesity consists of dieting and physical exercise.

 Diet programs may produce weight loss over the short term, but maintaining this weight loss is frequently difficult and often requires making exercise and a
lower calorie diet a permanent part of an individual's lifestyle.

 Success rates of long-term weight loss maintenance with lifestyle changes are low, ranging from 2 to 20%.

 Dietary and lifestyle changes are effective in limiting excessive weight gain in pregnancy and improve outcomes for both the mother and the child.

 The National Institutes of Health recommend a weight loss goal of 5% to 10% of the person's current weight over six months.
</item 3>

<item 4>
Joel Fuhrman
Joel Fuhrman (born December 2, 1953) is an American celebrity doctor who advocates what he calls a micronutrient-rich diet.
</item 4>

<item 5>
Eat Smart, Move More, Weigh Less
Eat Smart, Move More, Weigh Less is a 15-week adult weight management program that uses strategies based on evidence for weight loss and/or weight maintenance.

 The Eat Smart Move More Weigh Less classes focus on the 12 evidence-based eating and physical activity behaviors for weight management.

 The program does not provide a prescriptive diet plan, it teaches small lifetime changes.

 The program teaches mindfulness as a strategy to become more aware of eating and physical activity.

 Eat Smart, Move More, Weigh Less includes methods for planning and tracking healthy eating and physical activity behaviors.

 In 2007, Eat Smart, Move More, Weigh Less curriculum was developed by a writing team from NC State University and NC Division of Public Health.

 A complete listing of authors and their respective affiliations can be found on the program website.

 The 15-lesson curriculum was peer reviewed by state and local nutrition and physical activity professionals, representatives from the medical community, and a
CDC project officer.
</item 5>

<item 6>
Weight loss
Weight loss, in the context of medicine, health, or physical fitness, refers to a reduction of the total body mass, due to a mean loss of fluid, body fat or
adipose tissue or lean mass, namely bone mineral deposits, muscle, tendon, and other connective tissue.

 Weight loss can either occur unintentionally due to malnourishment or an underlying disease or arise from a conscious effort to improve an actual or perceived
overweight or obese state.

 "Unexplained" weight loss that is not caused by reduction in calorific intake or exercise is called cachexia and may be a symptom of a serious medical
condition.

 Intentional weight loss is commonly referred to as slimming.
</item 6>

<item 7>
Eat to Live
Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss is a book written in 2003 by Dr. Joel Fuhrman, which he revised in 2011.

 The book offers a formula: health = nutrients / calories.
</item 7>

<item 8>
Weight Wins
Weight Wins is a UK weight loss business which has developed and commercialised the concept of personal contracts for weight loss.

 The company has attracted attention from UK and international media for its ‘Pounds for Pounds’ weight loss programme and for operating the first trial by the
UK’s National Health Service (NHS) of a weight loss incentive programme.
</item 8>

<item 9>
Scarsdale diet
The Scarsdale diet is a fad diet designed for weight loss created in the 1970s by Herman Tarnower, named for the town in New York where he practiced cardiology,
described in the book "The Complete Scarsdale Medical Diet plus Dr. Tarnower's Lifetime Keep-Slim Program", which Tarnower wrote with an author of self-help
books, Sam Sinclair Baker.

 The diet carries potential health risks and does not instill the kind of healthy eating habits required for sustainable weight loss.
</item 9>

<item 10>
Very-low-calorie diet
Very low calorie diet (VLCD) is a diet with very or extremely low daily food energy consumption.

 It is defined as a diet of 800 kcal per day or less.

 VLCDs are formulated, nutritionally complete, liquid meals containing 800 kilocalories or less per day.

 VLCDs also contain the recommended daily requirements for vitamins, minerals, trace elements, fatty acids and protein.

 Carbohydrate may be entirely absent, or substituted for a portion of the protein; this choice has important metabolic effects.

 The VLCD products are usually a powder which is mixed with water or another low-food-energy liquid.

 The VLCD is prescribed on a case to case basis for rapid weight loss (about 1.5 to 2.5 kilograms or 3 to 5 pounds per week) in people with Body Mass Index of
30 and above.

 The health care provider can recommend the diet to a patient with a BMI between 27 and 30 if the medical complications the patient has due to being overweight
present serious health risks.

 It results in 4% more weight loss over the short term as compared to control.
</item 10>
=== END ITEM id=c94ebc224a822108 ===

=== ITEM id=3e0fcabb3639139e ===
Gold sentences and facts:
Sentence 0: Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records during three sessions in September and October 1954 and February 1955.
  Fact 0.0: Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records.
  Fact 0.1: Dean Martin recorded Swingin' Down Yonder during three sessions.
  Fact 0.2: The sessions for Swingin' Down Yonder took place in September 1954, October 1954 and February 1955.
Sentence 1:  One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless charisma and self-assurance.
  Fact 1.0: Dean Martin was one of the most popular and enduring American entertainers of the mid-20th century.
  Fact 1.1: Dean Martin was nicknamed the 'King of Cool'.
  Fact 1.2: Dean Martin's nickname 'King of Cool' reflected his seemingly effortless charisma.
  Fact 1.3: Dean Martin's nickname 'King of Cool' reflected his seemingly effortless self-assurance.

Retrieved items:
<item 1>
2120 South Michigan Ave.
2120 South Michigan Ave. is the fifteenth studio album by George Thorogood and the Destroyers.

 It was released on June 14, 2011 on the Capitol Records label.

 The album peaked at #2 on the "Billboard" Top Blues Albums chart.

 The title refers to the address of the offices and recording studios of Chess Records in Chicago.

 The album contains ten covers of songs recorded on Chess Records by artists such as Chuck Berry, Bo Diddley, Willie Dixon, and Muddy Waters; plus a cover of
The Rolling Stones' instrumental "2120 South Michigan Avenue" and two original songs about Chess Records artists.

 Capitol Records approached Thorogood with the idea for the album and selected most of the songs.

 Buddy Guy and Charlie Musselwhite perform on the album, although their work was added after primary recording was complete.
</item 1>

<item 2>
Willie and the Hand Jive
"Willie and the Hand Jive" is a song written by Johnny Otis and originally released as a single in 1958 by Johnny Otis Show, reaching #9 on the "Billboard" Hot
100 chart and #5 on the Billboard R&B chart.

 The song has a Bo Diddley beat and was partly inspired by the music sung by a chain gang Otis heard while he was touring.

 The lyrics are about a man who became famous for doing a dance with his hands, but the song has been accused of glorifying masturbation.

 It has since been covered by numerous artists, including The Strangeloves, Eric Clapton, Cliff Richard, Kim Carnes, George Thorogood and The Grateful Dead.

 Clapton's 1974 version was also released as a single and also reached the Billboard Top 40, peaking at #26.

 Thorogood's 1985 version reached #25 on the Hot Mainstream Rock Tracks chart.
</item 2>

<item 3>
Swingin' Down Yonder
Swingin' Down Yonder is the first full-length, 12-inch album recorded by Dean Martin for Capitol Records during three sessions in September and October 1954 and
February 1955.

 According to the original sleeve notes, all the songs have a "common geographical root: the American South."

 In 1963, Capitol Records re-released "Swingin' Down Yonder" under the titled "Southern Style".
</item 3>

<item 4>
Dean Martin Sings
Dean Martin Sings is the first studio album by Italian-American singer Dean Martin, released in 1953.

 It is the first long-play 10-inch album recorded by Martin for Capitol Records during two sessions recorded on the evening of November 20, 1952.

 The first session was recorded between 5 and 8 PM and it produced five songs featuring string arrangements.

 "There's My Lover" was recorded but not released.

 After a ninety-minute break, Martin was joined by a brass arrangement to record the remaining four songs.

 Seven of the eight songs on this album appeared in the Martin & Lewis film, "The Stooge".

 Two years later, the songs from this 10-inch album would be combined with four newly popular songs recorded between 1951 and 1953 to create a full-length
12-inch album.

 The 2005 Collectors' Choice reissue added four bonus songs recorded between 1949 and 1953 and was released with alternative cover artwork.
</item 4>

<item 5>
Collins &amp; Harlan
Collins & Harlan, the team of Arthur Collins and Byron G. Harlan formed a popular comic duet between 1903 and 1926.

 They sang ragtime standards as well as what were known as "Coon songs" – music sung by white performers in a black dialect.

 Their material also employed many other stereotypes of the time including Irishmen and farmers.

 Fellow recording artist Billy Murray nicknamed them "The Half-Ton Duo" as both men were rather overweight.

 Collins and Harlan produced many number one hits with recordings of minstrel songs such as "My Gal Irene", "I Know Dat I'll be Happy Til I Die", "Who Do You
Love?"

 and "Down Among the Sugarcane".

 Their song "That Funny Jas Band from Dixieland", recorded November 8, 1916, is among the first recorded uses of the word "jas" which eventually evolved to
"jass", and to the current spelling "jazz".
</item 5>

<item 6>
Bigtop Records
Bigtop Records was an American record label started by music executive Johnny Bienstock and the major music publisher Hill & Range Music and was co-owned along
with Big Top Record Distributors (sic).

 Hit artists included Del Shannon, Johnny and the Hurricanes, Lou Johnson, Sammy Turner, Don and Juan, and Toni Fisher.

 Big Top Record Distributors also distributed Paul Case's Dunes Records label in the early 60's, which had hits from Ray Peterson ("Corrina, Corrina") and
Curtis Lee ("Pretty Little Angel Eyes"), both records produced by Phil Spector.

 Bell Records briefly distributed Bigtop prior to the label closing, around 1965.

 "Bigtop" also released two Mad Magazine music parody themed albums; "Mad Twists Rock 'N' Roll" and "Fink Along With Mad" in 1963.
</item 6>

<item 7>
Lil Ru
Sylvester Samuels better known by his stage name Lil Ru, is an American rapper from Ridgeway, South Carolina currently signed to Def Jam Recordings.

 His debut album, 21 & Up was released on August 25, 2009.

 The Ridgeway native was 16 when he made his professional foray into the music business.

 Inspired by New Orleans’ innovative Cash Money Crew, Ru began making a name for himself on his local music scene, doing live shows and pressing up his own CDs.

 His hard-hitting lyrics and entrepreneurial spirit caught the attention of fellow South Carolinian Angie Stone.

 Shortly after the neo-soul songstress helped him secure a deal with Elektra Records, Ru found himself unsigned again, among the artists lost in the shuffle
after the label merged with Atlantic Records.

 Music fans first heard him on his 2001 debut single Will Destroy.

 He then released his 2002 follow up, Shawty What You Doin’.

 Both songs reached the Billboard R&B/Hip-Hop charts and helped land him at his next label home, Capitol Records.
</item 7>

<item 8>
One O'Clock Jump
"One O'Clock Jump" is a jazz standard, a 12-bar blues instrumental, written by Count Basie in 1937.

 The melody derived from band members' riffs--Basie rarely wrote down musical ideas, so Eddie Durham and Buster Smith helped him crystallize his ideas.

 The original 1937 recording of the tune by Basie and his band is noted for the saxophone work of Herschel Evans and Lester Young, trumpet by Buck Clayton,
Walter Page on bass and Basie himself on piano.
</item 8>

<item 9>
Cecil Gant
Cecil Gant (April 4, 1913 – February 4, 1951) was an American blues singer, songwriter and pianist, whose recordings of both ballads and "fiery piano rockers"
were successful in the mid- and late 1940s, and influenced the early development of rock and roll.

 His biggest hit was the 1944 ballad, "I Wonder".
</item 9>

<item 10>
Speedy West
Wesley Webb West (January 25, 1924 – November 15, 2003), better known as Speedy West, was an American pedal steel guitarist and record producer.

 He frequently played with Jimmy Bryant, both in their own duo and as part of the regular Capitol Records backing band for Tennessee Ernie Ford and many others.

 He also played on Loretta Lynn's first single.
</item 10>
=== END ITEM id=3e0fcabb3639139e ===
