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

=== ITEM id=6a6ae896b8de57b2 ===
Gold sentences and facts:
Sentence 0: Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil War.
  Fact 0.0: Am Rong was born in 1929.
  Fact 0.1: Am Rong died in May 1975.
  Fact 0.2: Am Rong was Cambodian.
  Fact 0.3: Am Rong was a soldier.
  Fact 0.4: Am Rong was a filmmaker.
  Fact 0.5: Am Rong acted as a spokesman on military matters for the Khmer Republic.
  Fact 0.6: Am Rong acted as a spokesman during the Cambodian Civil War.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is American.
  Fact 1.2: Ava Marie DuVernay is a director.
  Fact 1.3: Ava Marie DuVernay is a screenwriter.
  Fact 1.4: Ava Marie DuVernay is a film marketer.
  Fact 1.5: Ava Marie DuVernay is a film distributor.

Retrieved items:
<item 1>
Ava DuVernay
Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.

 At the 2012 Sundance Film Festival, DuVernay won the Best Director Prize for her second feature film "Middle of Nowhere", becoming the first African-American
woman to win the award.

 For her work in "Selma" (2014), DuVernay was the first black female director to be nominated for a Golden Globe Award.

 With "Selma", she was also the first black female director to have her film nominated for the Academy Award for Best Picture.

 In 2017, she was nominated for the Academy Award for Best Documentary Feature for her film "13th" (2016).
</item 1>

<item 2>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 2>

<item 3>
D. Ross Lederman
David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
</item 3>

<item 4>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 4>

<item 5>
List of accolades received by Selma (film)
"Selma" is a 2014 American historical drama film directed by Ava DuVernay, and produced by Dede Gardner, Jeremy Kleiner, Christian Colson, and Oprah Winfrey.

 The screenplay was written by Paul Webb.

 The film follows the events leading up to and during the Selma to Montgomery marches, and the resulting establishment of the Voting Rights Act of 1965 which
prohibited racial discrimination in voting in the United States.

 The film stars David Oyelowo as civil rights activist Martin Luther King Jr., Tom Wilkinson as President Lyndon B. Johnson, and Tim Roth as Governor of Alabama
George Wallace.

 Carmen Ejogo, Dylan Baker, Wendell Pierce, Common, and Winfrey feature in supporting roles.
</item 5>

<item 6>
Am Rong
Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil
War.

 Western journalists commented on the irony of his name as he gave briefings which "painted a rosy picture of the increasingly desperate situation on the
ground" during the war.
</item 6>

<item 7>
Middle of Nowhere (2012 film)
Middle of Nowhere is a 2012 independent feature film written and directed by Ava DuVernay and starring Emayatzy Corinealdi, David Oyelowo, Omari Hardwick and
Lorraine Toussaint.

 The film was the winner of the Directing Award for U.S. Dramatic Film at the 2012 Sundance Film Festival.
</item 7>

<item 8>
Emayatzy Corinealdi
Emayatzy Evett Corinealdi ( ; born January 14, 1980) is an American actress.

 She is best known for her leading role in the Ava DuVernay film "Middle of Nowhere" (2012).
</item 8>

<item 9>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 9>

<item 10>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 10>
=== END ITEM id=6a6ae896b8de57b2 ===

=== ITEM id=eeff881be44e4182 ===
Gold sentences and facts:
Sentence 0: Rynella is an unincorporated community in Iberia Parish, Louisiana, that consists of a few residential dwellings and a volunteer fire department along Highway 329 (also known as the Avery Island Highway).
  Fact 0.0: Rynella is an unincorporated community.
  Fact 0.1: Rynella is in Iberia Parish, Louisiana.
  Fact 0.2: Rynella consists of a few residential dwellings.
  Fact 0.3: Rynella has a volunteer fire department.
  Fact 0.4: The residential dwellings and the volunteer fire department are along Highway 329.
  Fact 0.5: Highway 329 is also known as the Avery Island Highway.
Sentence 1:  The name "Rynella" allegedly is an acronym deriving from the last two letters of the first names of three local women, Rosema"RY", Pauli"NE", and Lei"LA" McIlhenny (someone eventually adding an extra L to the resulting word).
  Fact 1.0: The name Rynella allegedly is an acronym.
  Fact 1.1: The acronym derives from the last two letters of the first names of three local women.
  Fact 1.2: The three local women are Rosema, Pauli, and Lei McIlhenny.
  Fact 1.3: The acronym uses the letters RY from Rosema, NE from Pauli, and LA from Lei.
  Fact 1.4: Someone eventually added an extra L to the resulting word.
Sentence 2:  They were daughters of Edward Avery McIlhenny, a noted Louisiana conservationist who presided over McIlhenny Company, maker of Tabasco brand pepper sauce at nearby Avery Island.
  Fact 2.0: The three women were daughters of Edward Avery McIlhenny.
  Fact 2.1: Edward Avery McIlhenny was a noted Louisiana conservationist.
  Fact 2.2: Edward Avery McIlhenny presided over McIlhenny Company.
  Fact 2.3: McIlhenny Company is the maker of Tabasco brand pepper sauce.
  Fact 2.4: McIlhenny Company is at nearby Avery Island.
Sentence 3: Tabasco sauce is a brand of hot sauce made exclusively from tabasco peppers ("Capsicum frutescens" var.
  Fact 3.0: Tabasco sauce is a brand of hot sauce.
  Fact 3.1: Tabasco sauce is made exclusively from tabasco peppers.
  Fact 3.2: The tabasco pepper is Capsicum frutescens var. tabasco.
Sentence 4:  "tabasco"), vinegar, and salt.
  Fact 4.0: Tabasco sauce is made with vinegar.
  Fact 4.1: Tabasco sauce is made with salt.

Retrieved items:
<item 1>
Nerine masoniorum
Nerine masoniorum is a species of flowering plant in the family Amaryllidaceae, subfamily Amaryllidoideae, native to the eastern Cape Province of South Africa.

 It is a bulbous perennial.

 "N. masoniorum" belongs to the group of nerines that have narrow evergreen foliage.

 The thread-like leaves reach a length of 25 cm or more.

 The flowering stem is 15–25 cm tall, with up to 11 flowers arranged in an umbel.

 Each flower has six narrow pink tepals with wavy edges.

 It flowers in late summer in cultivation, the first of the nerines to do so.
</item 1>

<item 2>
Sriracha sauce
Sriracha (Thai: ศรีราชา ,  ] ; ) is a type of hot sauce or chili sauce made from a paste of chili peppers, distilled vinegar, garlic, sugar, and salt.

 It is named after the coastal city of Si Racha, in Chonburi Province of eastern Thailand, where it may have been first produced for dishes served at local
seafood restaurants.
</item 2>

<item 3>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
</item 3>

<item 4>
Valentina (hot sauce)
Valentina is a brand of pourable hot sauce manufactured by Salsa Tamazula, a company in Guadalajara, Mexico.

 It is typically sold in 12.5-ounce and very large (one-liter or 34-ounce) glass bottles, with a flip-top cap permanently attached to the bottle (the cap does
not unscrew).

 The sauce, and the parent company's Tamazula hot sauce, are made with puya chilis from Jalisco state – similar to the Guajillo chili and known by the name
"guajillo puya" – is described as thicker than Tabasco sauce and less vinegary, with more chile flavor.

 Valentina comes in two varieties, hot and extra hot, and the sauce is known for its use on tacos and its taste, as opposed to only for its heat.

 Valentina's ingredients are water, chili peppers, vinegar, salt, spices and sodium benzoate (as a preservative).
</item 4>

<item 5>
Yucca rupicola
Yucca rupicola is a plant in the family Asparagaceae, known as the twistleaf yucca, twisted-leaf yucca, Texas yucca or twisted-leaf Spanish-dagger.

 The species was described by George Heinrich Adolf Scheele in 1850.

 This is a small, acaulescent plant with distinctive twisted leaves.

 It is native to the Edwards Plateau region of Texas and also to northeastern Mexico (Coahuila, Nuevo León).
</item 5>

<item 6>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 6>

<item 7>
Rapanea
Rapanea is a genus of plant in family Primulaceae.

 It has often been placed in synonymy with "Myrsine", and many species have been moved to "Myrsine".
</item 7>

<item 8>
Riverton, New York
Riverton, New York was a planned community built in 1973 in the town of Henrietta, New York, along the Genesee River.

 Riverton was seventh of the thirteen communities receiving Federal guarantee assistance from HUD as part of their New Communities Program created by the Urban
Growth and New Community Development Act of 1970.

 $12 million was earmarked for the community.

 Riverton was supposed to encompass a 2335 acre land area bordered on the north by the New York State Thruway, on the west by the Genesee River, on the east by
East River Rd., and on the south by the Rush-Henrietta town line, with some development continuing east along Erie Station Rd.

 Original plans called for construction to occur for a 16-year period and eventually house over 25,000 people, with the community including an 18-hole golf
course, 12 swimming pools, 3 artificial lakes, a marina on the Genesee, and commercial and industrial centers.

 However, only a small portion of this was built along Scottsville-West Henrietta Rd.
</item 8>

<item 9>
Tabasco sauce
Tabasco sauce is a brand of hot sauce made exclusively from tabasco peppers ("Capsicum frutescens" var.

 "tabasco"), vinegar, and salt.

 It is produced by McIlhenny Company of Avery Island, Louisiana.
</item 9>

<item 10>
Rynella, Louisiana
Rynella is an unincorporated community in Iberia Parish, Louisiana, that consists of a few residential dwellings and a volunteer fire department along Highway
329 (also known as the Avery Island Highway).

 The name "Rynella" allegedly is an acronym deriving from the last two letters of the first names of three local women, Rosema"RY", Pauli"NE", and Lei"LA"
McIlhenny (someone eventually adding an extra L to the resulting word).

 They were daughters of Edward Avery McIlhenny, a noted Louisiana conservationist who presided over McIlhenny Company, maker of Tabasco brand pepper sauce at
nearby Avery Island.

 (At one time McIlhenny operated a general store at Rynella and he owned much of the land that now comprises and surrounds the community.)
</item 10>
=== END ITEM id=eeff881be44e4182 ===

=== ITEM id=5f2534e33a266ddf ===
Gold sentences and facts:
Sentence 0:  Akrofuom is located in the Obuasi Municipal district of the Ashanti Region of Ghana.
  Fact 0.0: Akrofuom is located in the Obuasi Municipal district.
  Fact 0.1: The Obuasi Municipal district is in the Ashanti Region of Ghana.
Sentence 1: The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.
  Fact 1.0: The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.

Retrieved items:
<item 1>
Western Region (Ghana)
The Western Region is located in south Ghana, spreads from the Ivory Coast border in the west to the Central region in the east, includes the capital and large
twin city of Sekondi-Takoradi on the coast, coastal Axim, and a hilly inland area including Elubo.

 It includes Ghana's southernmost location, Cape Three Points, where crude oil was discovered in commercial quantities in June 2007.

 The Western Region enjoys a long coastline that stretches from South Ghana's border with Ivory Coast to the Western region's boundary with the Central Region
on the east.
</item 1>

<item 2>
Akrofuom (Ghana parliament constituency)
Akrofuom is one of the constituencies represented in the Parliament of Ghana.

 It elects one Member of Parliament (MP) by the first past the post system of election.

 Akrofuom is located in the Obuasi Municipal district of the Ashanti Region of Ghana.
</item 2>

<item 3>
Taifa, Accra
Taifa is a town in the Ga East Municipal District, a district in the Greater Accra Region of south-eastern Ghana near the capital Accra.

 Taifa is the twenty-sixth largest settlement in Ghana, in terms of population, with a population of 68,459 people.

 Taifa is located in the northwest suburbs area of Accra.

 It has a breakpoint on a railway line and a small park located on the northern edge of the location of the Taifa Ghana Atomic Energy Commission.

 At the Ghana 2000 census of 26 March 2000, the population was 26,145 inhabitants living in the city.

 Projections of 1 January 2007 estimated the population to be 48,927 inhabitants.

 In the census of 1984 there was only 1,009 inhabitants.

 The strong population growth of the Town is influenced by, among other things, a large number of illegal immigrants from west African countries who move to
towns and villages near the industrial town of Tema, just to find a job.
</item 3>

<item 4>
Tema
Tema is a city on the Bight of Benin and Atlantic coast of Ghana.

 It is located 25 km east of the capital city; Accra, in the region of Greater Accra, and is the capital of the Tema Metropolitan District.

 As of 2013, Tema is the eleventh most populous settlement in Ghana, with a population of approximately 161,612 people – a marked decrease from its 2005 figure
of 209,000.

 The Greenwich Meridian (00 Longitude) passes directly through the city.

 Tema is locally nicknamed the "Harbour Town" because of its status as Ghana's largest seaport.
</item 4>

<item 5>
Obuasi Municipal District
The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.

 The capital of the Obuasi Municipality is Obuasi.
</item 5>

<item 6>
Presbyterian University College
The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.

 It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.

 It was established by the Presbyterian Church of Ghana on 23 November 2003 and inaugurated on 27 March 2004 by the president of Ghana, John Agyekum Kufuor.
</item 6>

<item 7>
Techimantia
Techimantia is a city located in the Tano District of the Brong Ahafo Region of Ghana.

 It is one of the large cities in the region.

 It is believed to be originally inhabited by Hunters for the Ashanti King from Kumasi.

 These Hunters originally settled in the region after they found that there were lots of elephants in the region which could be hunted for the Ashanti King.The
name of the King is Oti Ampem.Currently,the chief of the town is no more(dead).

The occupation of the people in the town is large scale tomatoes farming.Techimantia is one of the largest towns in the region but there are few productive or
government work in the town.This was attributed to the curse from Okomfo Anokye of the Ashanti Kingdom.
</item 7>

<item 8>
Dodi Papase
Dodi Papase is a town in the Volta Region of Ghana.

 With a population of approximately 5,254 ,Dodi Papase is the second-most-populous city in the Volta region of Ghana.

 It is located in eastern Ghana on the banks of the Asukawkaw River and is known for the Dodi Papase Community Secondary School, a second cycle institution
(senior high school).
</item 8>

<item 9>
Ashanti Region
The Ashanti Region is located in south Ghana and is third largest of 10 administrative regions, occupying a total land surface of 24389 km2 or 10.2 per cent of
the total land area of Ghana.

 In terms of population, however, it is the most populated region with a population of 4,780,380 according to the 2010 census, accounting for 19.4% of Ghana’s
total population.

 The Ashanti Region is known for its major gold bar and cocoa production.

 The largest city and regional capital is Kumasi.
</item 9>

<item 10>
Tafo
Tafo is a town in Kumasi Metropolitan District in the Ashanti Region of Ghana near the regional capital Kumasi.

 Tafo is the thirtieth most populous settlement in Ghana, in terms of population, with a population of 60,919 people.

 Because of the town's population and housing development in recent years, it is debatable whether Tafo is still regarded as a separate town, or already a
suburb of Kumasi, the capital of the Ashanti region.

 The town is near Kumasi, with a distance of approximately 3.3 kilometers to the center of a similar name sounding village named New Tafo and must be
distinguished from Tafo.

 Tarkwa is located just 4.6 km away from Tafo.

 The city center of Kumasi is located approximately 9.8 kilometers away.

 Tafo is one of the urban constituencies of the Kumasi Metropolitan Assembly, the town's parliamentary candidate shall have one direct seat to the Parliament of
Ghana.
</item 10>
=== END ITEM id=5f2534e33a266ddf ===

=== ITEM id=e289a23207601824 ===
Gold sentences and facts:
Sentence 0: Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
  Fact 0.0: Hänt Extra is a Swedish language magazine.
  Fact 0.1: Hänt Extra is a weekly magazine.
  Fact 0.2: Hänt Extra is a celebrity and youth magazine.
  Fact 0.3: Hänt Extra is published by Aller Media.
  Fact 0.4: Hänt Extra is published in Stockholm, Sweden.
Sentence 1: Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
  Fact 1.0: Tea Moderna is a Macedonian woman's weekly magazine.
  Fact 1.1: Tea Moderna's name in Macedonian is Теа Модерна.
  Fact 1.2: Tea Moderna is a magazine for women.

Retrieved items:
<item 1>
Cult MTL
Cult MTL is an English language arts, culture and news website and monthly print publication, based in Montreal, Quebec, Canada.

 Its first print edition appeared on 7 September 2012.

 It was created only a few months after Montreal's last English-language alternative weekly, "Montreal Mirror", was unceremoniously closed by its parent
company, Quebecor.

 The founding editor's of "Cult MTL" were also involved with the "Mirror".

 In August 2013 the print version of the magazine was started.
</item 1>

<item 2>
2015 Macedonian Football Supercup
The 2015 Macedonian Supercup was the 3rd Macedonian Football Supercup, an annual Macedonian football match played between the winners of the previous season's
First League and Macedonian Cup.

 The game will be played between Rabotnički, who beat Teteks to win the 2015 Macedonian Cup Final, and Vardar, champions of the 2014–15 First League.

 It was played at Philip II Arena, Skopje and was won by Vardar by 4–3 on penalties after the match was ended 1–1 after the regular and extra time.
</item 2>

<item 3>
Football Weekly
Football Weekly is a podcast about football, hosted by "The Guardian" newspaper in the United Kingdom.

 Originally provided weekly (as its name suggests), its popularity led to a mid-weekly spinoff, "Football Weekly Extra".

 "Football Weekly" airs on Mondays during the European football season, after the weekend's fixtures, while "Football Weekly Extra" airs on Thursdays.

 During the Euro 2008, World Cup 2010, Euro 2012, World Cup 2014 and Euro 2016 tournaments the podcast aired daily.
</item 3>

<item 4>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 4>

<item 5>
Hänt Bild
Hänt Bild is a celebrity magazine published in Sweden.

 The magazine was started in 2003.

 It is part of Aller Media AB.

 It is published on a weekly basis.
</item 5>

<item 6>
Makedonsko Sonce
Makedonsko Sonce (Macedonian: ; English: Macedonian Sun ) is a weekly magazine published in the Republic of Macedonia.

 The title means "Macedonian Sun", referring to the Vergina Sun, which is used in the logo of the magazine.

 It was established by Gjorgija (George) Atanasoski and the first edition was published on June 24, 1994.

 It describes its own political stance as affirming "Macedonian national values, not only on territory of Macedonia but also in the other parts of ethnic
Macedonia".
</item 6>

<item 7>
Nikola Mladenov
Nikola Mladenov (Macedonian: Никола Младенов ; 10 March 1964 – 26 March 2013) was a Macedonian journalist, actor, founder and managing editor of the weekly
newspaper "Fokus", widely regarded as one of the most reputable journalists in Macedonia, who along with his colleagues from the newspaper "Mlad borec" is
considered to have laid the basis for the pluralistic principles in the Macedonian journalism.

 The "Award for best research story of the year – Nikola Mladenov", which is awarded by the Macedonian Institute for Media, is named after him.
</item 7>

<item 8>
Tea Moderna (magazine)
Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
</item 8>

<item 9>
Den Za Nas
Den Za Nas (in Macedonian: Ден за нас ) is the title of the fourth Macedonian language album by the Macedonian superstar Toše Proeski released in 2004.

 It is a 2-CD set which contains all of the songs performed on the National final in order to select a song for the Eurovision Song Contest 2004 with which
Proeski would represent Macedonia in the contest.

 The winning song was "Life" which qualified for the final and finished on the 14th place in the final.

 Some extra songs such as "Dzvezdo Severnice", "Me Sakaš So Zborovi", "Ima Li Den Za Nas" and others were recorded especially for the album.
</item 9>

<item 10>
Katarina Ivanovska
Katarina Ivanovska (Macedonian: Катарина Ивановска ; born 18 August 1988) is a Macedonian model and actress.

 She began her modeling career in 2004, appearing at Milan Fashion Week after winning the Look Models International model search in Macedonia.

 In December, 2004, she appeared in a pictorial for "Elle" magazine and has also appeared in "Citizen K", "Stiletto" and the Italian and Russian "Vogue".

 She has been featured on the covers of "Diva" and "Maxima" magazines and in advertisements for D&G in 2006.

 She is considered the most successful Macedonian model.

 In 2010, Ivanovska appeared in Serbian "Elle" magazine.

 In 2011 she signed a contract for advertising Victoria's Secret products.

 In 2011 she got her first acting job in the Macedonian World War II film, "The Third Half", landing the lead role of a young Jewish girl named Rebecca.
</item 10>
=== END ITEM id=e289a23207601824 ===

=== ITEM id=201f51ce14b04fae ===
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
The Blizzard (1923 film)
The Blizzard is a 1923 Swedish drama film directed by Mauritz Stiller, starring Einar Hanson, Mary Johnson, Pauline Brunius and Hugo Björne.

 The film's original Swedish title is Gunnar Hedes saga, which means "The story of Gunnar Hede".

 The narrative revolves a student who tries to save his family's mansion which is facing bankruptcy.

 The film is loosely based on the Selma Lagerlöf novel "The Tale of a Manor".
</item 1>

<item 2>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 2>

<item 3>
Actuality film
The actuality film is a non-fiction film genre that, like the documentary film, uses footage of real events, places, and things, yet unlike the documentary is
not structured into a larger argument, picture of the phenomenon or coherent whole.

 In practice, actuality films preceded the emergence of the documentary.

 During the era of early cinema, actualities—usually lasting no more than a minute or two and usually assembled together into a program by an exhibitor—were
just as popular and prominent as their fictional counterparts.

 The line between "fact" and "fiction" was not so sharply drawn in early cinema as it would become after the documentary came to serve as the predominant
non-fiction filmmaking form.

 An actuality film is not like a newspaper article so much as it is like the still photograph that is published along with the article, with the major
difference being that it moves.

 Apart from the traveling actuality genre, actuality is one film genre that remains strongly related to still photography.
</item 3>

<item 4>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 4>

<item 5>
Fire and Ice (1983 film)
Fire and Ice is a 1983 American adult animated epic high fantasy adventure film directed by Ralph Bakshi.

 The film, a collaboration between Bakshi and Frank Frazetta, was distributed by 20th Century Fox, which also distributed Bakshi's 1977 release, "Wizards".

 The animated feature, based on characters Bakshi and Frazetta co-created, was made using the process of rotoscoping, in which scenes were shot in live action
and then traced onto animation cels.
</item 5>

<item 6>
Return to Snowy River
Return to Snowy River is the original motion picture soundtrack from the 1988 film "The Man from Snowy River II".

 The soundtrack album is named after the American title for the film.
</item 6>

<item 7>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 7>

<item 8>
A Star Is Born (1976 film)
A Star Is Born is a 1976 American musical drama film telling the story of a young woman, played by Barbra Streisand, who enters show business, and meets and
falls in love with an established male star, played by Kris Kristofferson, only to find her career ascending while his goes into decline.

 It is a remake of two earlier versions – the 1937 version was a drama starring Janet Gaynor and Fredric March, and the 1954 version was a musical starring Judy
Garland and James Mason.

 It will be remade for a third time in 2018 starring Stefani Germanotta and Bradley Cooper.
</item 8>

<item 9>
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
</item 9>

<item 10>
Woman's film
The woman's film is a film genre which includes women-centered narratives, female protagonists and is designed to appeal to a female audience.

 Woman's films usually portray "women's concerns" such as problems revolving around domestic life, the family, motherhood, self-sacrifice, and romance.

 These films were produced from the silent era through the 1950s and early 1960s, but were most popular in the 1930s and 1940s, reaching their zenith during
World War II.

 Although Hollywood continued to make films characterized by some of the elements of the traditional woman's film in the second half of the 20th century, the
term itself disappeared in the 1960s.

 The work of directors George Cukor, Douglas Sirk, Max Ophüls, and Josef von Sternberg has been associated with the woman's film genre.

 Joan Crawford, Bette Davis, and Barbara Stanwyck were some of the genre's most prolific stars.
</item 10>
=== END ITEM id=201f51ce14b04fae ===

=== ITEM id=10b8a4c64a29f98f ===
Gold sentences and facts:
Sentence 0: Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).
  Fact 0.0: Blinding Edge Pictures is an American film production company.
  Fact 0.1: Blinding Edge Pictures was founded in 2000 by M. Night Shyamalan.
  Fact 0.2: Blinding Edge Pictures is known for producing films written and directed by Shyamalan.
  Fact 0.3: Blinding Edge Pictures produced Unbreakable (2000), written and directed by Shyamalan.
  Fact 0.4: Blinding Edge Pictures produced Signs (2002), written and directed by Shyamalan.
  Fact 0.5: Blinding Edge Pictures produced The Village (2004), written and directed by Shyamalan.
  Fact 0.6: Blinding Edge Pictures produced The Happening (2008), written and directed by Shyamalan.
  Fact 0.7: Blinding Edge Pictures produced After Earth (2013), written and directed by Shyamalan.
  Fact 0.8: Blinding Edge Pictures produced The Visit (2015), written and directed by Shyamalan.
  Fact 0.9: Blinding Edge Pictures produced Split (2017), written and directed by Shyamalan.
Sentence 1: Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson, alongside Robin Wright and Spencer Treat Clark.
  Fact 1.0: Unbreakable is a 2000 American superhero thriller film.
  Fact 1.1: Unbreakable was written by M. Night Shyamalan.
  Fact 1.2: Unbreakable was produced by M. Night Shyamalan.
  Fact 1.3: Unbreakable was directed by M. Night Shyamalan.
  Fact 1.4: Unbreakable stars Bruce Willis.
  Fact 1.5: Unbreakable stars Samuel L. Jackson.
  Fact 1.6: Robin Wright is in the cast of Unbreakable.
  Fact 1.7: Spencer Treat Clark is in the cast of Unbreakable.

Retrieved items:
<item 1>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 1>

<item 2>
Graph query row:
organization: Blinding Edge Pictures
production_relationship: CREATED_BY
creative_work: Unbreakable
Bruce_Willis: None
acting_relationship: None
</item 2>

<item 3>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 3>

<item 4>
Last Man Standing (1996 film)
Last Man Standing is a 1996 American action thriller film written and directed by Walter Hill and starring Bruce Willis, Christopher Walken and Bruce Dern.

 It is a credited remake of Akira Kurosawa's "Yojimbo".
</item 4>

<item 5>
Signs (film)
Signs is a 2002 American science fiction horror film written and directed by M. Night Shyamalan and executive produced by Shyamalan, Frank Marshall, Kathleen
Kennedy and Sam Mercer.

 A joint collective effort to commit to the film's production was made by Blinding Edge Pictures and The Kennedy/Marshall Company.

 It was commercially distributed by Touchstone Pictures theatrically, and by Touchstone Home Entertainment in home media format.

 Its story focuses on a former Episcopal priest named Graham Hess, played by Mel Gibson, who discovers a series of crop circles in his cornfield.

 Hess slowly discovers that the phenomena are a result of extraterrestrial life.

 It also stars Joaquin Phoenix, Rory Culkin, and Abigail Breslin.

 "Signs" explores faith, kinship, and extraterrestrials.
</item 5>

<item 6>
Dying of the Light (film)
Dying of the Light is a 2014 American psychological thriller film written and directed by Paul Schrader and starring Nicolas Cage, Anton Yelchin and Irène Jacob
about a government agent who must track down and kill a terrorist before he loses his full memory from a disease.

 It was released theatrically and through VOD formats by Lionsgate on December 5, 2014.

 The film received extremely negative reviews, with controversy surrounding the heavy tampering and reediting of the footage by the studio, who denied Schrader
final-cut privilege and led him and principal members of the cast to disown the released version and campaign against it.
</item 6>

<item 7>
Unbreakable (film)
Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson,
alongside Robin Wright and Spencer Treat Clark.

 The movie is the first installment in a trilogy.

 In "Unbreakable", a security guard named David Dunn survives a horrific train crash.

 After the incident, with the help of a manipulative disabled comic book shop owner named Elijah Price, he learns that he possesses superhuman powers.

 As Dunn explores and reluctantly confronts his powers while trying to navigate a difficult family life, he begins to fight crime and learns the true nature of
Elijah Price.
</item 7>

<item 8>
Blinding Edge Pictures
Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by
Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).

 In 2015, the company released its first television series "Wayward Pines".
</item 8>

<item 9>
Striking Distance
Striking Distance is a 1993 American action thriller film starring Bruce Willis as Pittsburgh Police homicide detective Thomas Hardy.

 The film co-stars Sarah Jessica Parker, Dennis Farina, and Tom Sizemore.

 It was directed by Rowdy Herrington and written by Herrington and Marty Kaplan.

 The film was shot on location throughout Pittsburgh; its early title was "Three Rivers".
</item 9>

<item 10>
Mercury Rising
Mercury Rising is a 1998 American political action thriller film starring Bruce Willis and Alec Baldwin.

 Directed by Harold Becker, the movie is based on Ryne Douglas Pearson's 1996 novel originally published as "Simple Simon".

 Willis plays Art Jeffries, an undercover FBI agent who protects a 9-year-old boy with autism who is targeted by government assassins after he cracks a top
secret government code.
</item 10>
=== END ITEM id=10b8a4c64a29f98f ===

=== ITEM id=4ea83d163281eef0 ===
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
</item 1>

<item 2>
Escape Tour
The Escape Tour was a concert tour by the American rock band Journey.

 It was one of Journey's most successful concert tours, in support of their first number one album, "Escape".

 The tour included six consecutive sold out dates at the Pine Knob Theatre in Detroit, and four straight sold out shows at the Los Angeles Forum, and Chicago's
Rosemont Horizon.

 Journey also made an appearance on July 2, 1982 at the Rose Bowl in Pasadena, California with Blue Öyster Cult, Triumph and Aldo Nova.

 The 8 ⁄ -month tour took Journey through Japan and North America.

 Point Blank, Billy Squier, the Greg Kihn Band and Loverboy were the opening acts during the tour.
</item 2>

<item 3>
U2 360° Tour
The U2 360° Tour was a worldwide concert tour by rock band U2.

 Launched in support of the group's 2009 album "No Line on the Horizon", the tour visited stadiums from 2009 through 2011.

 The concerts featured the band playing "in the round" on a circular stage, allowing the audience to surround them on all sides.

 To accommodate the stage configuration, a large four-legged structure nicknamed "The Claw" was built above the stage, with the sound system and a cylindrical,
expanding video screen on top of it.

 At 164 ft tall, it was the largest stage ever constructed.

 U2 claimed that the tour would be "the first time a band has toured in stadiums with such a unique and original structure."
</item 3>

<item 4>
Electrical Storm (song)
"Electrical Storm" is a song by Irish rock band U2.

 It was released as a single from their second compilation album, "The Best of 1990–2000", and was one of the two new songs recorded for that album (the other
one being "The Hands That Built America").

 The music video for the song prominently features drummer Larry Mullen Jr., as well as actress Samantha Morton.

 The song was written by Bono and is about two lovers who are fighting and the tension between them; it relates this to a looming electrical storm.

 U2 did not play the song live until 2 July 2009, when they performed it on the second concert of their U2 360° Tour in Barcelona.
</item 4>

<item 5>
Psycho Circus World Tour
They used a translator on stage to let the crowd understand exactly what Stanley was saying.

 At the end of the performance, the band ignited all of the pyrotechnics at once; as a result, they were banned from performing in Bremen.

 Ticket sales for this tour were notably slower than the previous Reunion Tour, with many of the smaller market shows underselling and a second North American
leg for the summer of 1999 cancelled all together, the band ultimatly decided on embarking on a Farewell tour in the new millennium.
</item 5>

<item 6>
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
</item 6>

<item 7>
The Year of the Voyager
The Year Of The Voyager is a double DVD/CD released by Nevermore.

 It was released in Europe on October 20, 2008 and in North America on November 25, 2008 via Century Media.

 The set covers the "This Godless Endeavor" touring cycle, starting with live footage from the U.S. Gigantour 2005, the Metal Mania festival 2006 in Poland, the
Wacken Open Air festival in 2006 in Germany and the main DVD show recorded at the Zeche in Bochum, Germany.

 Bonus material includes two songs from Century Media USA 10th Anniversary Party 2001, all promo videos and an interview with singer Warrel Dane which was
recorded at the Roax Film Studios in Berlin in the spring of 2008.
</item 7>

<item 8>
Clearview 77
The opening track on their 2000 EP "Another Story" produced by Agnostic Front frontman Roger Miret was played on K-ROCK's "The Buzz" with Matt Pinfield.

 This led to many 1st's for the band as they performed on live TV for "Studio Y" on cables Metro Channel.

 Clearview 77 also received ample airplay on national radio with their cover of the U2 classic "With or Without You" in 2002.

 At the tail end of their career the band recorded 2 songs and filmed TV commercial spots for the cable network TNN.

 The promo spots for the networks James Bond marathon aired on MTV, ESPN, TNN, TNT, TV Land, and Nickolodeon in the winter of 2002.

 The band parted ways and retired in the beginning of 2004.
</item 8>

<item 9>
Ceremonials Tour
The Ceremonials Tour was the second concert tour by English indie rock band Florence and the Machine.

 The tour included performances at Music festivals because it is lead singer Florence Welch's favourite way to perform live.

 Welch had originally planned to spend over one year touring for "Ceremonials" (their previous tour lasted almost three and a half years) before announcing that
the December 2012 dates would be the final performances of the tour.

 On 25 February 2013 it was announced that Florence and the Machine would be playing at the Coke Live Music Festival in Poland on 10 August 2013, and on 26
March 2013 it was announced that the band would be playing at Chime for Change's "The Sound of Change Live" concert at London's Twickenham Stadium alongside
Beyoncé, Ellie Goulding and Haim, amongst others.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=4ea83d163281eef0 ===

=== ITEM id=84f6c0b55df1aadb ===
Gold sentences and facts:
Sentence 0: Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on science and technology subjects.
  Fact 0.0: Popular Science is also known as PopSci.
  Fact 0.1: Popular Science is an American bi-monthly magazine.
  Fact 0.2: Popular Science carries popular science content.
  Fact 0.3: Popular science content refers to articles for the general reader on science and technology subjects.
Sentence 1: Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.
  Fact 1.0: Girlfriends was a women's magazine.
  Fact 1.1: Girlfriends provided critical coverage of culture.
  Fact 1.2: Girlfriends provided critical coverage of entertainment.
  Fact 1.3: Girlfriends provided critical coverage of world events.
  Fact 1.4: Girlfriends' coverage was from a lesbian perspective.
Sentence 2:  Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.
  Fact 2.0: Girlfriends was published monthly from San Francisco since 1993.
  Fact 2.1: Girlfriends was distributed nationwide by Disticor.
Sentence 3:  "Girlfriends" magazine ceased publication in 2006.
  Fact 3.0: Girlfriends magazine ceased publication in 2006.

Retrieved items:
<item 1>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 1>

<item 2>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 2>

<item 3>
Popular Science
Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on
science and technology subjects.

 "Popular Science" has won over 58 awards, including the American Society of Magazine Editors awards for its journalistic excellence in both 2003 (for General
Excellence) and 2004 (for Best Magazine Section).

 With roots beginning in 1872, "Popular Science" has been translated into over 30 languages and is distributed to at least 45 countries.
</item 3>

<item 4>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 4>

<item 5>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 5>

<item 6>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 6>

<item 7>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 7>

<item 8>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 8>

<item 9>
Popular science
Popular science (also pop-science or popsci) is interpretation of science intended for a general audience.

 While science journalism focuses on recent scientific developments, popular science is more broad-ranging.

 It may be written by professional science journalists or by scientists themselves.

 It is presented in many forms, including books, film and television documentaries, magazine articles, and web pages.
</item 9>

<item 10>
ST/Amiga Format
ST Amiga Format was a computer magazine that covered the Atari ST and Amiga computers.

 It was published by Future plc to cover the ever growing market for the, then-new, 16-bit home computers.

 Issues were equally balanced with coverage for both Commodore Amiga and Atari ST systems.

 Issue 1 included a main feature 'ST or Amiga?

 The Choice is Yours', where the pros and cons for each machine was examined.
</item 10>
=== END ITEM id=84f6c0b55df1aadb ===

=== ITEM id=11da040524602220 ===
Gold sentences and facts:
Sentence 0: Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.
  Fact 0.0: Ted Kooshian was born October 8, 1961.
  Fact 0.1: Ted Kooshian is an American jazz pianist.
  Fact 0.2: Ted Kooshian is an American jazz keyboardist.
  Fact 0.3: Ted Kooshian is an American jazz composer.
  Fact 0.4: Ted Kooshian has performed with Aretha Franklin.
  Fact 0.5: Ted Kooshian has performed with Chuck Berry.
  Fact 0.6: Ted Kooshian has performed with Marvin Hamlisch.
  Fact 0.7: Ted Kooshian has performed with Sarah Brightman.
  Fact 0.8: Ted Kooshian has performed with Il Divo.
  Fact 0.9: Ted Kooshian has performed with Blood, Sweat, and Tears.
Sentence 1:  Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.
  Fact 1.0: Marvin Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.
Sentence 2:  This collection of all four is referred to as an "EGOT".
  Fact 2.0: The collection of all four awards (Emmy, Grammy, Oscar and Tony) is referred to as an 'EGOT'.

Retrieved items:
<item 1>
Lonesome Pine Fiddlers
This most talented family of musicians were the best, surpassed by none.

 Finally, on October 1, 2009, The Lonesome Pine Fiddlers got their due when they were inducted into the International Bluegrass Hall of Fame at the Ryman
Theater (the original home of the Grand Ole Opry).

 Bobby Osborne, Melvin Goins and Paul Williams were there to receive the bands award.

 In the crowd of a sold out theater was the son of Ezra Cline, Scotty Ireland Cline, who recalled being in that same theater as a child sitting on stage and
watching the Fiddlers play.

 (At the time, the Opry had bleachers for family just off stage).

 The final act of the evening at the IBMA Awards was the playing of "Pain in my Heart" by Osborne, Goins and Williams along with a Song from the Dillards, who
were also inducted the same evening.
</item 1>

<item 2>
List of James Tiptree Jr. Award winners
The following is a list of winners and shortlisted works of the James Tiptree Jr.

 Award, an annual literary prize for works of science fiction or fantasy that expand or explore one's understanding of gender.

 It was initiated in February 1991 by science fiction authors Pat Murphy and Karen Joy Fowler, and is awarded and discussed yearly at WisCon.
</item 2>

<item 3>
Peter C. Doherty
Peter Charles Doherty, (born 15 October 1940) is an Australian veterinary surgeon and researcher in the field of medicine.

 He received the Albert Lasker Award for Basic Medical Research in 1995, the Nobel Prize in Physiology or Medicine jointly with Rolf M. Zinkernagel in 1996 and
was named Australian of the Year in 1997.

 In the Australia Day Honours of 1997, he was named a Companion of the Order of Australia for his work with Zinkernagel.

 He is also a National Trust Australian Living Treasure.

 In 2009 as part of the Q150 celebrations, Doherty's immune system research was announced as one of the Q150 Icons of Queensland for its role as an iconic
"innovation and invention".
</item 3>

<item 4>
Marvin Hamlisch
Marvin Frederick Hamlisch (June 2, 1944August 6, 2012) was an American composer and conductor.

 Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.

 This collection of all four is referred to as an "EGOT".

 He is one of only two people (along with Richard Rodgers) to have won those four prizes and a Pulitzer Prize.
</item 4>

<item 5>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 5>

<item 6>
Toshi (musician)
Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and
songwriter.

 He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.

 Toshi then went on to have an extensive solo career.

 In 2007, X Japan reunited and began an ongoing world tour in 2010.

 His distinctive, impassioned tenor vocals have been compared to those of Steve Perry.
</item 6>

<item 7>
Philip K. Dick Award
The Philip K. Dick Award is a science fiction award given annually at Norwescon sponsored by the Philadelphia Science Fiction Society and (since 2005) supported
by the Philip K. Dick Trust, and named after science fiction and fantasy writer Philip K. Dick.

 It has been awarded since 1983, the year after Dick's death.

 Works that have received the award are identified on their covers as "Best Original SF Paperback".

 They are awarded to the best original paperback published each year in the US.
</item 7>

<item 8>
Triple Crown of Acting
The Triple Crown of Acting is a term used in the American entertainment industry to describe actors who have won a competitive Academy Award, Emmy Award, and
Tony Award in the acting categories.

 23 people have achieved the triple crown of acting (14 women, 9 men).

 Helen Hayes' Emmy Award win on February 5, 1953, made her the first person to achieve the triple crown.

 Thomas Mitchell became the first man to achieve the triple crown with his Tony Award win later the same year on March 29, 1953.

 Hayes and Rita Moreno are the only triple crown winners in competitive acting categories who have also won a Grammy Award to complete the EGOT.
</item 8>

<item 9>
List of awards and nominations received by Kim Tae-yeon
List of awards and nominations received by Kim Tae-yeon
</item 9>

<item 10>
Tatsuhiro Oshiro
Tatsuhiro Ōshiro (大城 立裕 , Ōshiro Tatsuhiro , born 19 September 1925) is a Japanese novelist and playwright.

 He was awarded the Akutagawa Prize in 1967 for his novella of the same year, "The Cocktail Party", which has been adapted for the stage and made into a film.

 Ōshiro has also been an innovator of the traditional Ryukyuan narrative dance form known as kumi odori.

 Having added twenty new pieces to the repertoire, Ōshiro is credited as having "single-handedly revived the genre that originated in the 18th century" by
incorporating Okinawa shibai (dramas in the Okinawan language) and distinctive rhythms to construct a fluid, hybrid cultural identity.

 His writings have been noted for making Okinawan culture and history accessible to Japanese readership, while his more popular works have been critically
praised for "offering an acute perspective on the psychological and moral implications of war and military occupation."
</item 10>
=== END ITEM id=11da040524602220 ===

=== ITEM id=b90b830f32982d29 ===
Gold sentences and facts:
Sentence 0:  He played three seasons for the Philadelphia Eagles (1940) and the Pittsburgh Steelers (1941–1942).
  Fact 0.0: The player played three seasons for the Philadelphia Eagles.
  Fact 0.1: The player's Philadelphia Eagles seasons are listed as 1940.
  Fact 0.2: The player played for the Pittsburgh Steelers.
  Fact 0.3: The player's Pittsburgh Steelers seasons are listed as 1941-1942.
Sentence 1:  He was drafted by the Philadelphia Eagles in the second round of the 2008 NFL Draft, and played for the Washington Redskins for three seasons after his departure from the Eagles.
  Fact 1.0: The player was drafted by the Philadelphia Eagles.
  Fact 1.1: The player was drafted in the second round of the 2008 NFL Draft.
  Fact 1.2: The player played for the Washington Redskins for three seasons.
  Fact 1.3: The player played for the Washington Redskins after leaving the Eagles.

Retrieved items:
<item 1>
1986 Tampa Bay Buccaneers season
The 1986 Tampa Bay Buccaneers season began with the team trying to improve on an 2–14 season.

 This was one of the worst seasons in franchise history.

 There is some sentiment that the 1986 team was even worse than the winless team of 1976, and the 473 points conceded was not beaten by any NFL team until the
2001 Indianapolis Colts gave up 486.

 The Buccaneers selected Bo Jackson with the top pick in the draft, but were unable to convince him to join the team.

 Three weeks after the draft, Jackson signed a three-year baseball contract with the Kansas City Royals.

 Despite holding four of the first forty selections in the draft, and the presence of a great influx of fresh talent from defunct USFL teams, the Buccaneers
were unable to find any impact players in either the draft or free agency.

 They entered the season with a roster nearly identical to the previous season's 2–14 team.
</item 1>

<item 2>
Brooklyn Dodgers (NFL)
Later co-owner and partner Dan Topping (1912–1974), pulled the Tigers team out of the old NFL in 1946 and placed it in the newly established rival professional
league – the All-America Football Conference, which shortly lasted until 1949 until several stronger teams from the AAFC merged with and entered a reorganized
NFL in 1950.

 It lasted until 1970 with the NFL-AFL (third) merger following the establishment of the first "Super Bowl" inter-league national championship game three years
before with the old NFL champions playing the victors of the latest rival fourth incarnation of the American Football League IV, formed in 1960 (now the
American Football Conference (AFC).
</item 2>

<item 3>
Verlon Biggs
Verlon Marion Biggs (March 16, 1943 – June 7, 1994) was an American football defensive end in the American Football League and National Football League.

 He played for the New York Jets (AFL) in Super Bowl III, but felt he didn't receive enough credit for the Jets' playoff win against the Oakland Raiders in the
AFL Championship Game that launched them into the Super Bowl.

 He played well but sulked until 1970, demanded more money, and wound up signing with Vince Lombardi's Washington Redskins of the NFL.

 Always a dominating defensive end, Verlon solidified George Allen's defense (Allen replaced Lombardi upon his death in July 1970) and led the Redskins into
Super Bowl VII.

 His nickname with the Redskins was "dirty Biggs" because of his extremely physical style of play.
</item 3>

<item 4>
The entries profile NFL/CFL players, coaches, executives, and records. Jim O’Brien won Super Bowl V for Baltimore with a last-second field goal; Marc Mueller
coaches Calgary, and Johnny Bright and George Reed are CFL legends. Damon Allen retired with 72,381 passing yards. The NFL’s seven-member 2,000-yard club began
with O.J. Simpson; Eric Dickerson holds the record (2,105). Philadelphia’s 2010 “Miracle at the New Meadowlands”
</item 4>

<item 5>
Brooklyn Dodgers (NFL)
The Brooklyn Dodgers were an American football team that played in the National Football League from 1930 to 1943, and in 1944 as the Brooklyn Tigers.

 The team played its home games at Ebbets Field of the baseball National League's team, the Brooklyn Dodgers.

 In 1945, because of financial difficulties and the increasing scarcity of major league-level players because of the war-time defense requirements at the height
of World War II, the team was merged with the Boston Yanks and were known as the Yanks for that season.

 This old NFL franchise was not related to the earlier (second incarnation) American Football League II with a franchise that played as the Brooklyn Tigers for
the first half of the 1936 season before moving to Rochester, New York and playing as the Rochester Tigers.

 Another NFL team that played in the Brooklyn borough was the Brooklyn Lions (which became the Brooklyn Horsemen after merging with a team from an earlier first
incarnation AFL of the same name) in 1926.
</item 5>

<item 6>
DeSean Jackson
DeSean William Jackson (born December 1, 1986) is an American football wide receiver for the Tampa Bay Buccaneers of the National Football League (NFL).

 He played college football for the University of California, Berkeley, where he was recognized as a consensus All-American.

 He was drafted by the Philadelphia Eagles in the second round of the 2008 NFL Draft, and played for the Washington Redskins for three seasons after his
departure from the Eagles.

 Jackson has been selected to the Pro Bowl three times, and was the first player selected to the Pro Bowl at two different positions in the same year when he
was named to the 2010 Pro Bowl as a wide receiver and return specialist.
</item 6>

<item 7>
Dallas Texans (NFL)
The Dallas Texans played in the National Football League (NFL) for one season, 1952, with a record of 1–11.

 The team is considered one of the worst teams in NFL history, both on (lowest franchise winning percentage) and off the field.

 The team was based first in Dallas, then Hershey, Pennsylvania, and Akron, Ohio, during its only season.

 The Texans were the last NFL team to fold.

 Many players on the 1952 roster went to the new Baltimore Colts franchise in 1953.

 The American Football League (AFL) had a 1960 charter member named the Dallas Texans (who later became the Kansas City Chiefs), but the AFL Texans have no
relationship with the earlier NFL team.
</item 7>

<item 8>
Don Looney
John Don Looney (September 2, 1916 – April 5, 2015) was a professional American football end in the National Football League.

 He was born in Sulphur Springs, Texas.

 He played three seasons for the Philadelphia Eagles (1940) and the Pittsburgh Steelers (1941–1942).

 He was the first receiver in NFL history to have over 100 yards receiving in each of his first two games, a feat which was not equaled until the 2008 NFL
season by another Eagles wide receiver, DeSean Jackson.

 At the time of his death, Looney was the second oldest living former NFL player.

 He was the father of NFL running back Joe Don Looney, who later died in a one-person motorcycle accident after his NFL career ended.
</item 8>

<item 9>
Sylvester Croom, now Titans running-backs coach, was Mississippi State’s 2004–08 head coach and the SEC’s first Black head coach. The 1976 Rams won the NFC West
at 10–3–1, beat Dallas, then lost to Minnesota; the 1986 Buccaneers went 2–14, conceded 473 points and failed to sign top draft pick Bo Jackson. In 2010,
Philadelphia erased a 21-point deficit against the Giants, winning
</item 9>

<item 10>
Wilbert Montgomery
Wilbert Montgomery (born September 16, 1954) is a former American football player in the National Football League for nine years with the Philadelphia Eagles
and Detroit Lions.

 In the past, Montgomery has been the running backs and tight ends coach for St. Louis Rams (1997–2005), the running backs coach for the Detroit Lions
(2006-2007), the running backs coach for the Baltimore Ravens (2008–2013), and the running backs coach for the Cleveland Browns (2014–2015).
</item 10>
=== END ITEM id=b90b830f32982d29 ===

=== ITEM id=b1da2810a8e142b5 ===
Gold sentences and facts:
Sentence 0:  The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
  Fact 0.0: The breed is used for hunting in packs.
  Fact 0.1: The breed descends from the old Hound of Saintonge type of large hunting dog.
Sentence 1: The Kritikos Lagonikos (Cretan Hound) (Greek:Kρητικός Λαγωνικός) is a hunting breed of dog from the island of Crete, in Greece.
  Fact 1.0: The Kritikos Lagonikos, also called the Cretan Hound, is a hunting breed of dog.
  Fact 1.1: The Kritikos Lagonikos has the Greek name Κρητικός Λαγωνικός.
  Fact 1.2: The Kritikos Lagonikos is from the island of Crete, in Greece.

Retrieved items:
<item 1>
Braque Francais
The Braques français are hunting dogs, from a very old type of gun dog used for pointing the location of game birds for a hunter.

 There are two breeds of Braque français, both from the south of France, the Braque français, type Gascogne (French Pointing Dog - Gascogne type, larger size)
and the Braque français, type Pyrénées (French Pointing Dog - Pyrenean type, smaller size) They are popular hunting dogs in France, but are seldom seen
elsewhere.
</item 1>

<item 2>
Ariegeois
The Ariegeois is a breed of dog from the "département" of Ariège in the Midi-Pyrenées region of southern France.

 It is a medium-sized pack-hunting scenthound deriving from crossing of Grand Bleu de Gascogne and Grand Gascon-Saintongeois hounds with local Briquet dogs.

 It is used both as a courser and for driving game to waiting guns.

 While most successful with hares, it is also used for hunting deer and boar.

 It is distinguished by its friendly nature with other hounds and affection for human companions.
</item 2>

<item 3>
Chien Français Blanc et Orange
The Chien Français Blanc et Orange (FCI No.316) is a breed of dog of the scenthound type, originating in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 3>

<item 4>
Griffon Bleu de Gascogne
The Griffon Bleu de Gascogne (FCI No.32) Is a breed of dog of the scenthound type, originating in France, and is a versatile hunting dog, used on small and
large game, in packs or individually.

 The Griffon Bleu de Gascogne has a speckled, rough coat.
</item 4>

<item 5>
Petit Bleu de Gascogne
The Petit Bleu de Gascogne (FCI No.31) Is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog.

 The Petit Bleu de Gascogne is not a small (petite) dog, the name comes from its use on small game.
</item 5>

<item 6>
Grand Anglo-Français Blanc et Noir
The Grand Anglo-Français Blanc et Noir (FCI No.323) translated into English as the Great Anglo-French White and Black Hound, is a breed of dog used in hunting
as a scenthound, usually in packs.

 It is one of the Anglo-French hound breeds which were created by crossing French scenthounds with English (Anglo) Foxhounds.
</item 6>

<item 7>
Grand Bleu de Gascogne
The Grand Bleu de Gascogne (FCI No.22) is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog, and is an important breed in the ancestry of many other hounds.
</item 7>

<item 8>
Chien Français Tricolore
The Chien Français Tricolore (FCI No.219) translated into English as the French Tricolour Hound, is a breed of dog of the scenthound type, originating in
France.

 The breed is used for hunting in packs.
</item 8>

<item 9>
Grand Anglo-Français Blanc et Orange
The Grand Anglo-Français Blanc et Orange (FCI No.324) translated into English as the Great Anglo-French White and Orange Hound, is a breed of dog used in
hunting as a scenthound, usually in packs.

 It is one of the Anglo-French hound breeds which were created by crossing French scenthounds with English (Anglo) foxhounds.
</item 9>

<item 10>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 10>
=== END ITEM id=b1da2810a8e142b5 ===
