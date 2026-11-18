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

=== ITEM id=7e1f3eaf9ae748fa ===
Gold sentences and facts:
Sentence 0:  His siblings were Abby, Nelson, Laurance, Winthrop, and David.
  Fact 0.0: Abby was his sibling.
  Fact 0.1: Nelson was his sibling.
  Fact 0.2: Laurance was his sibling.
  Fact 0.3: Winthrop was his sibling.
  Fact 0.4: David was his sibling.
Sentence 1: David Rockefeller (June 12, 1915 – March 20, 2017) was an American banker who was chairman and chief executive of Chase Manhattan Corporation.
  Fact 1.0: David Rockefeller lived from June 12, 1915 to March 20, 2017.
  Fact 1.1: David Rockefeller was an American banker.
  Fact 1.2: David Rockefeller was chairman of Chase Manhattan Corporation.
  Fact 1.3: David Rockefeller was chief executive of Chase Manhattan Corporation.

Retrieved items:
<item 1>
David Rockefeller
David Rockefeller (June 12, 1915 – March 20, 2017) was an American banker who was chairman and chief executive of Chase Manhattan Corporation.

 He was the oldest living member of the Rockefeller family and family patriarch from August 2004 until his death in March 2017.

 Rockefeller was a son of John D. Rockefeller Jr. and Abby Aldrich Rockefeller, and a grandson of John D. Rockefeller and Laura Spelman Rockefeller.
</item 1>

<item 2>
Hope Aldrich Rockefeller
Hope Aldrich Rockefeller (born May 17, 1938) is an American newspaper publisher.

 She is the eldest daughter of philanthropist John Davison Rockefeller III (1906–1978) and Blanchette Ferry Hooker (1909–1992).

 Her elder brother is former Senator John Davison "Jay" Rockefeller IV (born 1937).
</item 2>

<item 3>
William B. Harrison Jr.
William B. Harrison Jr., born August 12, 1943, in Rocky Mount, North Carolina, is the former CEO and chairman of JPMorgan Chase.

 He attended high school at Virginia Episcopal School, where he was a basketball star.

 He attended the University of North Carolina at Chapel Hill, where he was admitted to the Zeta Psi fraternity.

 Having risen through the ranks of Chemical Bank before succeeding Walter V. Shipley during the Chemical Bank takeover of the Chase Manhattan Corporation, he
and Douglas A. Warner III, then CEO of J.P. Morgan & Co., were the principal architects of the US$30.9 billion Chase and J.P. Morgan & Co. merger of 2000.

 Harrison has been a director of the Firm or a predecessor institution since 1991.

 Harrison is also a director of Merck & Co., Inc.
</item 3>

<item 4>
JPMorgan Chase
JPMorgan Chase & Co. is a U.S. multinational banking and financial services holding company headquartered in New York City.

 It is the largest bank in the United States, the world's sixth largest bank by total assets, with total assets of US$2.5 trillion, and the world's second most
valuable bank by market capitalization, after ICBC.

 It is a major provider of financial services, and according to "Forbes" magazine is the world's sixth largest public company based upon a composite ranking.

 The hedge fund unit of JPMorgan Chase is the second largest hedge fund in the United States.

 The company was formed in 2000, when Chase Manhattan Corporation merged with J.P. Morgan & Co.
</item 4>

<item 5>
Rockefeller Brothers Fund
The Rockefeller Brothers Fund (RBF) is a philanthropic foundation created and run by members of the Rockefeller family.

 It was founded in New York City in 1940 as the primary philanthropic vehicle for the five third-generation Rockefeller brothers: John D. Rockefeller III,
Nelson, Laurance, Winthrop and David.

 It is distinct from the Rockefeller Foundation.

 The Rockefellers are an industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the late 19th and
early 20th centuries.
</item 5>

<item 6>
John D. Rockefeller III
John Davison Rockefeller III (March 21, 1906 – July 10, 1978) was a philanthropist and third-generation member of the prominent Rockefeller family.

 He was the eldest son of philanthropists John D. Rockefeller Jr. and Abby Aldrich Rockefeller.

 His siblings were Abby, Nelson, Laurance, Winthrop, and David.
</item 6>

<item 7>
Walter V. Shipley
Walter V. Shipley (born November 2, 1935) was the former Chairman and Chief Executive Officer of Chase Manhattan Bank and, before that, the company with which
it merged Chemical Bank.

 Shipley was named chief executive of Chemical in 1981 and held the position through 1999 and remained at the bank as chairman through January 2000, just prior
to the bank's merger with J.P. Morgan & Co..

 During his 18-year tenure, Shipley oversaw Chemical's mergers with Texas Commerce Bank in 1987, Manufacturers Hanover in 1991 and Chase Manhattan Bank in 1996.
</item 7>

<item 8>
Rockefeller family
The Rockefeller family ( ) is an American industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the
late 19th and early 20th centuries, with John D. Rockefeller and his brother William Rockefeller primarily through Standard Oil.

 The family is also known for its long association with and control of Chase Manhattan Bank.

 They are considered to be one of the most powerful families, if not the most powerful family, in the history of the United States.
</item 8>

<item 9>
Chase Bank
JPMorgan Chase Bank, N.A., doing business as Chase Bank, is a national bank that constitutes the consumer and commercial banking subsidiary of the U.S.
multinational banking and financial services holding company, JPMorgan Chase.

 The bank was known as Chase Manhattan Bank until it merged with J.P. Morgan & Co. in 2000.

 Chase Manhattan Bank was formed by the merger of the Chase National Bank and The Manhattan Company in 1955.

 The bank has been headquartered in Columbus, Ohio since its merger with Bank One Corporation in 2004.

 The bank acquired the deposits and most assets of Washington Mutual.
</item 9>

<item 10>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 10>
=== END ITEM id=7e1f3eaf9ae748fa ===

=== ITEM id=4b5d80d5a74f3f18 ===
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
Anigozanthos manglesii
Anigozanthos manglesii, commonly known as the red-and-green kangaroo paw, Mangles kangaroo paw, Kurulbrang(Noongar) is a plant species endemic to Western
Australia, and the floral emblem of that state.
</item 1>

<item 2>
Anigozanthos humilis
Anigozanthos humilis is a species of "Anigozanthos" in the family Haemodoraceae, known as common cat's paw or catspaw.

 This flowering perennial plant is endemic to Southwest Australia and widespread in its open forests.
</item 2>

<item 3>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 3>

<item 4>
Anigozanthos preissii
Anigozanthos preissii, the Albany cat's paw, is a herbaceous plant species in the family Haemodoraceae, endemic to Western Australia.
</item 4>

<item 5>
Leycesteria formosa
Leycesteria formosa (Himalayan honeysuckle, flowering nutmeg, Himalaya nutmeg or pheasant berry) is a deciduous shrub in the family Caprifoliaceae, native to
the Himalaya and southwestern China.

 It is considered a noxious invasive species in Australia, New Zealand, the neighboring islands of Micronesia, and some other places.
</item 5>

<item 6>
Mulga Island
Mulga Island is a small island 3 nmi off the coast and 5 nmi northeast of Kirkby Head, Enderby Land in Antarctica.

 Plotted from air photos taken from ANARE (Australian National Antarctic Research Expeditions) aircraft in 1956.

 Mulga is the vernacular name for species of Acacia found in semi-desert areas of Australia.
</item 6>

<item 7>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 7>

<item 8>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 8>

<item 9>
Anigozanthos
Anigozanthos is a small genus of Australian plants in the Bloodwort family Haemodoraceae.

 The 11 species and several subspecies are commonly known as kangaroo paw and catspaw depending on the shape of their flowers.

 A further species, previously identified as "Anigozanthos fuliginosus" and commonly known as the black kangaroo paw, has been transferred to its own monotypic
genus and is now known as "Macropidia fuliginosa".
</item 9>

<item 10>
Anigozanthos flavidus
Anigozanthos flavidus is a species of plant found in Southwest Australia.

 It is member of the Haemodoraceae family.

 It is commonly known as the tall, yellow, or evergreen, kangaroo paw.

 The specific epithet, "flavidus", refers to the yellow flowers of this plant.
</item 10>
=== END ITEM id=4b5d80d5a74f3f18 ===

=== ITEM id=562899da95322791 ===
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
Flaming Moe's
"Flaming Moe's" is the tenth episode of "The Simpsons"' third season.

 It first aired on the Fox network in the United States on November 21, 1991.

 In the episode, Homer tells Moe Szyslak of a secret alcoholic cocktail that includes cough medicine and fire that he calls "Flaming Homer".

 Moe steals the recipe from Homer, renames the drink the "Flaming Moe" and begins selling it at his tavern.

 The drink is a success and boosts business and patronage, but Homer is angry at Moe for his betrayal, and seeks revenge.
</item 2>

<item 3>
Man-eating tree
Man-eating tree can refer to any of the various legendary carnivorous plants large enough to kill and consume a person or other large animal.

 The carnivorous plant with the largest known traps is probably Nepenthes rajah, which produces pitchers up to 41 cm tall with a volume up to 3.5 l .

 The pitcher of this species are designed to trap arthropods.

 However, the same bait may also attract rodents like the summit rat ("Rattus baluensis") and the Mountain treeshrew ("Tupaia montana").

 Only very rarely will the rodents fall into the large pitchers of this species.

 Other large carnivorous plants that have similar properties include Nepenthes robcantleyi and Nepenthes attenboroughi.
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
Hotlum Glacier
The Hotlum Glacier is a glacier situated on the northeast flank of Mount Shasta, in the US state of California.

 It is the largest and most voluminous glacier in California, although not as thick or long as the nearby Whitney Glacier.

 The Hotlum Glacier flows from a large cirque on the northeast side of Mount Shasta's main summit below the Hotlum Headwall at roughly 13600 ft .

 It flows northeastward down the steep slope, forming three lobes which terminate near 10400 ft .
</item 9>

<item 10>
Flaming beverage
Flaming beverages include cocktails and other mixed drinks that contain flammable, high-proof alcohol, which is ignited prior to consumption.

 The alcohol may be an integral part of the drink, or it may be floated as a thin layer across the top of the drink.

 The flames are mostly for dramatic flair.

 However, in combination with certain ingredients, the flavor of the drink is altered.

 Some flavors are enhanced, and it may impart a toasted flavor to some drinks.
</item 10>
=== END ITEM id=562899da95322791 ===

=== ITEM id=6fb23798171b4641 ===
Gold sentences and facts:
Sentence 0: Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.
  Fact 0.0: Elizabeth Austin was born in 1958.
  Fact 0.1: Elizabeth Austin is an American writer.
  Fact 0.2: Elizabeth Austin lives in Oak Park, Illinois.
Sentence 1:  She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".
  Fact 1.0: Elizabeth Austin has also written for The Washington Monthly.
  Fact 1.1: Elizabeth Austin has also written for Ladies Home Journal.
  Fact 1.2: Elizabeth Austin has also written for Good Housekeeping.
  Fact 1.3: Elizabeth Austin has also written for Self.
  Fact 1.4: Elizabeth Austin has also written for Shape.
Sentence 2: Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight fitness) and became the number one women's fitness magazine.
  Fact 2.0: Shape is a monthly English language fitness magazine.
  Fact 2.1: Shape was started by Weider Publications in 1981.
  Fact 2.2: Shape was founded by Christine MacIntyre.
  Fact 2.3: Christine MacIntyre is a pioneer in women's free weight fitness.
  Fact 2.4: Shape became the number one women's fitness magazine.

Retrieved items:
<item 1>
Football World
Football World, later renamed "Athletic World", was an American magazine devoted to the coverage of inter-collegiate sports.

 Its masthead described it as "A Magazine With a Mission to Serve the College Man," a publication "devoted to Inter-collegiate Athletics and sports of Amateur
standing only."

 It was founded in 1921 by J. D. Fetzer.

 The name of the magazine was later changed to "Athletic World" as the coverage extended to a broader range of sports, including women's swimming.

 Unlike other sports magazines of the era, which focused on promoting a healthy lifestyle, "Football World/Athletic World" celebrated the entertainment value of
sports with a special emphasis on the personalities of famous athletes.

 The magazine was renamed "Outing" in December 1924, reflecting a change in its focus.
</item 1>

<item 2>
Wendy Austin
Dr Wendy Elizabeth Austin Hewitt, Wendy Austin is a BBC journalist and broadcaster in BBC Northern Ireland, and former presenter of BBC Radio Ulster’s current
affairs programme "Talkback".
</item 2>

<item 3>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 3>

<item 4>
Muscle &amp; Fitness (British magazine)
Muscle & Fitness is a British fitness magazine, which contains articles on strength and fitness training, diet plans and strategies for men and women, and tips
and advice.
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
Memphis Sport
Memphis Sport is a sports and fitness magazine featuring articles on local and regional teams, players and events.

 In addition, "Memphis Sport" focuses on health and fitness for an active lifestyle.

 The magazine debuted in Memphis, Tennessee in July 2006.

 Local franchises covered include the University of Memphis Tigers, Memphis Redbirds, Mississippi Riverkings and Tennessee Titans.
</item 6>

<item 7>
Ingrid Romero
Ingrid Romero (born July 10, 1985) is an international fitness model, fitness competitor, and fitness coach.

 Training & Fitness Magazine called her "one of the most recognized faces in the fitness industry," and "one of the most accomplished female fitness stars
ever."
</item 7>

<item 8>
Graph query row:
entity: Shape
relationship: CREATED_BY
related: Christine MacIntyre
</item 8>

<item 9>
Anais Zanotti
Anais Zanotti (born January 8, 1986 in Saint-Tropez) is a French model and stunt woman of French and Italian ancestry who lives in Miami, Florida.

 She has been featured in magazines such as "Playboy", "FHM", "Maxim", "GQ", "Cosmopolitan", "Vogue", and "People", Men's Fitness magazine, Muscle and Fitness,
Women Fitness magazine and more.

 She has over 40 magazine covers.

 Zanotti is also a certified skydiver with over 1250 jumps.GS .
</item 9>

<item 10>
Elizabeth Austin (writer)
Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.

 Austin has lived in the Midwestern United States all her life.

 She began her journalism career at the now-defunct City News Bureau of Chicago and has written articles for national news magazines such as "Newsweek" and
"Time", in addition to a wide range of other national and regional publications, such as "People", and the "Chicago Tribune".

 She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".

 She has often written about medicine and health care.

 Austin also has been heard on WBEZ, the National Public Radio station in Chicago, as well as the BBC in the UK and Scotland.

 She has appeared on the Today Show and on ITV in Europe.
</item 10>
=== END ITEM id=6fb23798171b4641 ===

=== ITEM id=abd25b0ab1ade93f ===
Gold sentences and facts:
Sentence 0:  The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova Planta Decree of Majorca and Ibiza (28 November 1715).
  Fact 0.0: The body that was to become the Gran i General Consell was founded in 1249.
  Fact 0.1: The Consell was abolished on 22 July 1718.
  Fact 0.2: The Consell was abolished by Philip V.
  Fact 0.3: Philip V was empowered by the Nova Planta Decree of Majorca and Ibiza.
  Fact 0.4: The Nova Planta Decree of Majorca and Ibiza is dated 28 November 1715.
Sentence 1: Philip V (Spanish: "Felipe V" , French: "Philippe" , Italian: "Filippo" ; 19 December 1683 – 9 July 1746) was King of Spain from 1 November 1700 to 15 January 1724, when he abdicated in favour of his son Louis, and from 6 September 1724, when he reassumed the throne upon his son's death, to his own death 9 July 1746.
  Fact 1.0: Philip V's Spanish name is Felipe V.
  Fact 1.1: Philip V's French name is Philippe.
  Fact 1.2: Philip V's Italian name is Filippo.
  Fact 1.3: Philip V was born on 19 December 1683.
  Fact 1.4: Philip V died on 9 July 1746.
  Fact 1.5: Philip V was King of Spain from 1 November 1700 to 15 January 1724.
  Fact 1.6: Philip V abdicated in favour of his son Louis on 15 January 1724.
  Fact 1.7: Philip V reassumed the Spanish throne on 6 September 1724, upon his son's death.
  Fact 1.8: Philip V was King of Spain from 6 September 1724 until his death on 9 July 1746.

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
House of Hanover
The House of Hanover (or the Hanoverians ; German: "Haus Hannover" ) is a German royal dynasty that ruled the Electorate and then the Kingdom of Hanover, and
that also provided monarchs of Great Britain and Ireland from 1714 and ruled the United Kingdom until the death of Queen Victoria in 1901.

 Upon Victoria's death, the British throne passed to her eldest son Edward VII, a member of the House of Saxe-Coburg and Gotha through his father.
</item 2>

<item 3>
Viscount Galway
Viscount Galway is a title that has been created once in the Peerage of England and thrice in the Peerage of Ireland.

 The first creation came in the Peerage of England in 1628 in favour of Richard Burke, 4th Earl of Clanricarde.

 He was made Earl of St Albans at the same time (see the Earl of Clanricarde for more information on this creation).

 The second creation came in the Peerage of Ireland in 1687 in favour of Ulick Bourke.

 He was made Baron Tyaquin at the same time, also in the Peerage of Ireland.

 However, both titles became extinct on his early death in 1691.

 The third creation came in the Peerage of Ireland in 1692 in favour of the French soldier and diplomat Henry de Massue, Marquis de Ruvigny.

 He was made Earl of Galway in 1697.

 However, both titles became extinct on his death in 1720.
</item 3>

<item 4>
Philip V of Spain
Philip V (Spanish: "Felipe V" , French: "Philippe" , Italian: "Filippo" ; 19 December 1683 – 9 July 1746) was King of Spain from 1 November 1700 to 15 January
1724, when he abdicated in favour of his son Louis, and from 6 September 1724, when he reassumed the throne upon his son's death, to his own death 9 July 1746.
</item 4>

<item 5>
Succession to the British throne
Succession to the British throne is determined by descent, gender (for people born before October 2011), legitimacy, and religion.

 Under common law, the Crown is inherited by a sovereign's children or by a childless sovereign's nearest collateral line.

 The Bill of Rights 1689 and the Act of Settlement 1701, restrict succession to the throne to the legitimate Protestant descendants of Sophia of Hanover that
are in "communion with the Church of England".

 Spouses of Roman Catholics were disqualified from 1689 until the law was amended in 2015.

 Protestant descendants of those excluded for being Roman Catholics are eligible.
</item 5>

<item 6>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 6>

<item 7>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 7>

<item 8>
Gran i General Consell
The Gran i General Consell (Catalan: "Great and General Council") was the supreme political, administrative, and representative organ of the Kingdom of Majorca.

 Since the Kingdom of Majorca did not have courts, the Gran i General Consell took over most of the functions they would otherwise have had, including the role
of a representative body.

 The Gran i General Consell evolved from the Catalan municipal councils, especially that of the City of Majorca (present-day Palma de Mallorca.

 The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova
Planta Decree of Majorca and Ibiza (28 November 1715).

 At the same time, all other separate Majorcan institutions were dissolved.
</item 8>

<item 9>
Duke of Bedford
He regained the earldom in 1485 when his nephew Henry VII came to the throne and was elevated to the dukedom the same year.

 He had no legitimate children and the titles became extinct on his death in 1495.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=abd25b0ab1ade93f ===

=== ITEM id=2e950419fb2d4a41 ===
Gold sentences and facts:
Sentence 0: Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All Stars, as well of his own solo band, the Kuwata band.
  Fact 0.0: Keisuke Kuwata's name is written 桑田 佳祐 (Kuwata Keisuke).
  Fact 0.1: Keisuke Kuwata was born February 26, 1956.
  Fact 0.2: Keisuke Kuwata is a Japanese multi-instrumentalist.
  Fact 0.3: Keisuke Kuwata is a Japanese singer-songwriter.
  Fact 0.4: Keisuke Kuwata is frontman for the Southern All Stars.
  Fact 0.5: Keisuke Kuwata has his own solo band, the Kuwata band.
Sentence 1: Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.
  Fact 1.0: Moya Brennan was born Máire Ní Bhraonáin.
  Fact 1.1: Moya Brennan is also known as Máire Brennan.
  Fact 1.2: Moya Brennan was born 4 August 1952.
  Fact 1.3: Moya Brennan is an Irish folk singer.
  Fact 1.4: Moya Brennan is an Irish songwriter.
  Fact 1.5: Moya Brennan is an Irish harpist.
  Fact 1.6: Moya Brennan is an Irish philanthropist.

Retrieved items:
<item 1>
Bruce Yonemoto
Bruce Yonemoto (born 1949) is a Japanese-American multimedia artist.

 His work has been recognized by the National Endowment for the Arts, the American Film Institute, The Rockefeller Foundation, the Maya Deren Award for
Experimental Film and Video, and a mid-career survey show at the Japanese American National Museum, in addition to major solo exhibitions at the
InterCommunication Center in Tokyo, the Institute of Contemporary Art the University of Pennsylvania, and the Kemper Museum in Kansas City, MO.

 Yonemoto was featured in the 2002 Corcoran Gallery Bienniel, in Washington D.C. Yonemoto was also a recipient of the Creative Capital Visual Arts Award in
2008.
</item 1>

<item 2>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 2>

<item 3>
Kevin Mayer
Kevin Mayer (] , ] or ] , born 10 February 1992) is a French athlete specializing in combined events.

 A native of Drôme in southeast France, Mayer excelled in combined events from a young age; highlights from his youth career include winning gold medals in the
octathlon at the 2009 World Youth Championships and in the decathlon at the 2010 World Junior Championships and at the 2011 European Athletics Junior
Championships, establishing a new national junior record at the latter.
</item 3>

<item 4>
My Match Is a Makin'
My Match Is A Makin' (English translation of opening track title) is a music album by Irish musicians Moya Brennan and Cormac de Barra.

 This is Moya's eighth studio album to be released.

 It was released on 14 April 2010 exclusively to concert goers on her Spring 2010 tour of the Netherlands.

 The album was re-released to a wider audience under the title Voices & Harps.
</item 4>

<item 5>
Keiko Utoku
Keiko Utoku (宇徳敬子 , "Utoku Keiko" , born April 7, 1967 in Izumi, Kagoshima, Japan) is a Japanese singer and songwriter under UK Sweet label.Keiko made first
debut with chorus part in band B.B.Queens then joined with J-pop female band Mi-Ke from 1991 to 1993, before starting a solo career.

 As a child, Utoku listened to a lot of music, including music made by Momoe Yamaguchi.

 She has written and sung many songs, such as "Wasurenagusa", and Sonic the Hedgehog fans remember her for her vocal work in the original version of the Sonic
CD video game's opening and closing themes in Japan and Europe, "Sonic – You Can Do Anything" and "Cosmic Eternity – Believe In Yourself" in 1993.

 In addition, she also provided Sonic's vocals in the game.

 At recent dates, she has released 15 singles, 5 digital singles, 4 studio albums, 1 mini and 2 best albums.
</item 5>

<item 6>
Moya Brennan
Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.

 She is the older sister of Enya and Brídín Brennan.

 She began performing professionally in 1970 when her family formed the band Clannad, and is considered as the "First Lady of Celtic Music".

 Moya released her first solo album in 1992 called "Máire", a successful venture.

 She has been nominated for two Grammys and has won an Emmy Award.

 She has recorded music for several soundtracks, including "Titanic", "To End All Wars" and "King Arthur".
</item 6>

<item 7>
2014 Monterrey Open – Doubles
Tímea Babos and Kimiko Date-Krumm were the defending champions, but chose not to participate together.

 Babos played alongside Olga Govortsova, while Date-Krumm teamed up with Karolína Plíšková.

 The two teams were scheduled to meet in the semifinals, but Date-Krumm withdrew with a right leg injury.

<br>
</item 7>

<item 8>
Kate O'Beirne
Kate Walsh O'Beirne (September 23, 1949 – April 23, 2017) was the President of National Review Institute.

 She was the Washington editor of "National Review".

 Her column, "Bread and Circuses," covered Congress, politics, and U.S. domestic policy.
</item 8>

<item 9>
Two Horizons
Two Horizons is the fifth solo album by Irish singer Moya Brennan.

 It is her first full-length release under the name "Moya Brennan", as opposed to "Máire Brennan".

 The album was predominantly recorded in her home studio in Dublin and was nominated for a Grammy award.

 The album was recorded between 2002–2003 and first became available on 23 October 2003.

 It is also Brennan's first non-Christian album since her 1993 recording Misty Eyed Adventures.

 It is the most successful of her solo albums to date and the second to be Grammy-nominated.
</item 9>

<item 10>
Koi Kaze
Koi Kaze (Japanese: 恋風 , lit.

 "Love Wind") is a Japanese manga series written and illustrated by Motoi Yoshida.

 It premiered in the September 2001 issue of "Evening" and ran for 29 chapters until its conclusion in the October 2004 issue.

 The individual chapters were collected and published in five "tankōbon" volumes by Kodansha.

 The series tells of the love affair that develops between 27-year-old Koshiro, and his 15-year-old sister, Nanoka.
</item 10>
=== END ITEM id=2e950419fb2d4a41 ===

=== ITEM id=c4e79837eaca83a2 ===
Gold sentences and facts:
Sentence 0:  The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss' 1983 book "Fatal Vision".
  Fact 0.0: The journalist in question is Joe McGinniss.
  Fact 0.1: The murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald.
  Fact 0.2: Dr. Jeffrey R. MacDonald became the subject of Joe McGinniss' 1983 book 'Fatal Vision'.
Sentence 1: Jeffrey Robert MacDonald (born October 12, 1943) is an American medical doctor who was convicted in 1979 of murdering his pregnant wife and two daughters in February 1970.
  Fact 1.0: Jeffrey Robert MacDonald was born on October 12, 1943.
  Fact 1.1: Jeffrey Robert MacDonald is an American medical doctor.
  Fact 1.2: Jeffrey Robert MacDonald was convicted in 1979 of murdering his pregnant wife.
  Fact 1.3: Jeffrey Robert MacDonald was convicted in 1979 of murdering his two daughters.
  Fact 1.4: The murders took place in February 1970.

Retrieved items:
<item 1>
Jeffrey R. MacDonald
Jeffrey Robert MacDonald (born October 12, 1943) is an American medical doctor who was convicted in 1979 of murdering his pregnant wife and two daughters in
February 1970.
</item 1>

<item 2>
Joe McGinniss
Joseph R. McGinniss, Sr. (December 9, 1942 – March 10, 2014), known as Joe McGinniss, was an American non-fiction writer and novelist.

 The author of twelve books, he first came to prominence with the best-selling "The Selling of the President 1968" which described the marketing of
then-presidential candidate Richard Nixon.

 He is popularly known for his trilogy of bestselling true crime books — "Fatal Vision", "Blind Faith" and "Cruel Doubt" — which were adapted into TV miniseries
in the 1980s and 90s.

 His last book was "The Rogue: Searching for the Real Sarah Palin", an account of Sarah Palin, the former governor of Alaska who was the 2008 Republican
vice-presidential nominee.
</item 2>

<item 3>
The Journalist and the Murderer
The Journalist and the Murderer is a study by Janet Malcolm about the ethics of journalism, published by Alfred A. Knopf/Random House in 1990.

 It is an examination of the professional choices that shape a work of non-fiction, as well as a rumination on the morality that underpins the journalistic
enterprise.

 The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss'
1983 book "Fatal Vision".
</item 3>

<item 4>
Fatal Vision (disambiguation)
Fatal Vision is the 1983 true crime book by Joe McGinniss which lies at the center of the "Fatal Vision" controversy.
</item 4>

<item 5>
Convicted (1950 film)
Convicted is a 1950 American crime film noir directed by Henry Levin starring Glenn Ford and Broderick Crawford.

 It was the third Columbia Pictures film adaptation of the 1929 stage play "The Criminal Code" by Martin Flavin, following Howard Hawk's "The Criminal Code"
(1931) and John Brahm's "Penitentiary" (1938).
</item 5>

<item 6>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 6>

<item 7>
Fatal Vision controversy
The controversy over Fatal Vision, journalist and author Joe McGinniss's best-selling 1983 true crime book, is a decades-long dispute spanning several court
cases and discussed in several other published works.
</item 7>

<item 8>
Fatal Vision (miniseries)
Fatal Vision is a 1984 American television miniseries based on the account, in the book of the same name, of the murders in 1970 at Fort Bragg of the wife and
daughters of U.S. Army officer Jeffrey R. MacDonald.
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
Going to Extremes (book)
Going to Extremes is a non-fiction book by Joe McGinniss.

 It was first published in 1980.

 The book is about McGinniss' travels through Alaska for a year.

 The book became a best-seller.
</item 10>
=== END ITEM id=c4e79837eaca83a2 ===

=== ITEM id=f9de25aaa232aece ===
Gold sentences and facts:
Sentence 0: The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.
  Fact 0.0: The Cathedral Church of the Holy Spirit, Guildford is an Anglican cathedral.
  Fact 0.1: The Cathedral Church of the Holy Spirit, Guildford is located at Guildford, Surrey, England.
Sentence 1: St Paul's Cathedral, London, is an Anglican cathedral, the seat of the Bishop of London and the mother church of the Diocese of London.
  Fact 1.0: St Paul's Cathedral, London is an Anglican cathedral.
  Fact 1.1: St Paul's Cathedral, London is the seat of the Bishop of London.
  Fact 1.2: St Paul's Cathedral, London is the mother church of the Diocese of London.

Retrieved items:
<item 1>
Lincoln Cathedral
Lincoln Cathedral or the Cathedral Church of the Blessed Virgin Mary of Lincoln, and sometimes St. Mary's Cathedral in Lincoln, England is the seat of the
Anglican bishop.

 Building commenced in 1088 and continued in several phases throughout the medieval period.

 It was the tallest building in the world for 238 years (1311–1549), and the first building to hold that title after the Great Pyramid of Giza.

 The central spire collapsed in 1549 and was not rebuilt.

 The cathedral is the third largest in Britain (in floor area) after St Paul's and York Minster, being 484 by .

 It is highly regarded by architectural scholars; the eminent Victorian writer John Ruskin declared: "I have always held... that the cathedral of Lincoln is out
and out the most precious piece of architecture in the British Isles and roughly speaking worth any two other cathedrals we have."
</item 1>

<item 2>
Guildford Cathedral, designed by Edward Maufe and built 1936–1961 on land donated by Lord Onslow, is the Church of England Diocese of Guildford’s seat. Its
exterior features sculpture by notable mid-century artists. Dean Dianna Gwilliams leads its chapter; Lanesborough School is its choir school. Barry Rose
established its choir and daily sung worship, while organist Katherine Dienes is the first woman to hold a Church of England cathedral’s senior musical post.
The
</item 2>

<item 3>
The entries cover British and Australian church heritage. London’s St Paul’s, founded on the site of a church dating to 604, is Wren’s Grade I-listed Baroque
cathedral, rebuilt after the Great Fire. Rockhampton’s St Paul’s Cathedral (1883–1953) and adjacent hall (built circa 1900) are Queensland-listed. Augustine
became Canterbury’s first archbishop in 597. Other cathedrals include medieval Lichfield (England’s only
</item 3>

<item 4>
Exterior sculpture of Guildford Cathedral
The Exterior sculpture of Guildford Cathedral provides many artistic features, including sculptures, engravings and more by some of England’s finest sculptors
and craftsmen of the 1950s and 1960s.

 The people who worked on the cathedral include: Edward Maufe, Alan Collins, Vernon Hill, Eric Gill, John Hutton, Dennis Huntley and others.
</item 4>

<item 5>
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
</item 5>

<item 6>
St Paul's Cathedral
St Paul's Cathedral, London, is an Anglican cathedral, the seat of the Bishop of London and the mother church of the Diocese of London.

 It sits on Ludgate Hill at the highest point of the City of London and is a Grade 1 listed building.

 Its dedication to Paul the Apostle dates back to the original church on this site, founded in AD 604.

 The present cathedral, dating from the late 17th century, was designed in the English Baroque style by Sir Christopher Wren.

 Its construction, completed in Wren's lifetime, was part of a major rebuilding programme in the City after the Great Fire of London.
</item 6>

<item 7>
St Philip's Cathedral, Birmingham
The Cathedral Church of Saint Philip is the Church of England cathedral and the seat of the Bishop of Birmingham.

 Built as a parish church and consecrated in 1715, St Philip's became the cathedral of the newly formed Diocese of Birmingham in 1905.

 St Philip's was built in the early 18th century in the Baroque style by Thomas Archer and is located on Colmore Row, Birmingham, England.

 The cathedral is a Grade I listed building.

 St Philip's is the third smallest cathedral in England after Derby and Chelmsford.
</item 7>

<item 8>
St Paul's Cathedral, Rockhampton
St Paul's Anglican Cathedral is a heritage-listed cathedral at 89 William Street, Rockhampton, Rockhampton Region, Queensland, Australia.

 It was designed principally by Annersley Voysey and built from 1883 to 1953.

 It is also known as St Paul's Church of England and St Paul's Church of England Cathedral.

 It was added to the Queensland Heritage Register on 21 October 1992.

 The associated hall and offices were listed on the Queensland Heritage Register on 23 June 2000.
</item 8>

<item 9>
Guildford Cathedral
The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.

 The Lord Onslow donated the land on which the Cathedral was built with a covenant for this purpose.

 Designed by Sir Edward Maufe and built between 1936 and 1961, it is the seat of the Diocese of Guildford.
</item 9>

<item 10>
Diocese of Guildford
The Diocese of Guildford is a Church of England diocese covering nine of the eleven districts in Surrey, much of north-east Hampshire and a parish in Greater
London.

 The cathedral is Guildford Cathedral and the bishop is the Bishop of Guildford.

 Of the two provinces of the church it falls within the Province of Canterbury.
</item 10>
=== END ITEM id=f9de25aaa232aece ===

=== ITEM id=c197edfab128c03d ===
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
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 1>

<item 2>
George W. Macfarlane
George Walter Hunter Macfarlane (March 1, 1849 – February 20, 1921) was a British businessman, courtier and politician of the Kingdom of Hawaii.

 He served Colonel of the military staff of King Kalākaua, traveling with the monarch on his 1881 world tour.

 He also served as his final chamberlain of king and was at his deathbed in 1891.
</item 2>

<item 3>
Charles Hastings Judd
Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.

 He served as Chamberlain and Colonel of the military staff of King Kalākaua and traveled with the monarch on his 1881 world tour.
</item 3>

<item 4>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
</item 4>

<item 5>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 5>

<item 6>
William K. MacNulty
William Kirk MacNulty (May 22, 1892 - August 3, 1964) was a U. S. Marine with a long and distinguished record.

 He was a young U.S. Marine Corps Second Lieutenant during World War One and saw action at the Battle of the Argonne Forest.

 He served as a captain during the Second U.S. Nicaraguan Campaign (1926–33).

 During the Second World War as Lieutenant Colonel he commanded the U.S. Marine Corps defense of Guam against overwhelming Imperial Japanese forces during the
First Battle of Guam.

 He was incarcerated by the Japanese as a prisoners of war.

 He was promoted to Brigadier General during captivity and retired from military service in 1946.

 He is buried at the Golden Gate National Cemetery in San Bruno, San Mateo County, California.
</item 6>

<item 7>
Merrie Monarch Festival
The Merrie Monarch Festival is a week-long cultural festival that takes place annually in Hilo, Hawaii during the week after Easter.

 It honors King David Kalākaua, who was called the "Merrie Monarch" for his patronage of the arts and is credited with restoring many Hawaiian cultural
traditions during his reign, including the hula.

 Many "hālau hula" (schools), including some from the U.S. mainland and some international performers, attend the festival each year to participate in
exhibitions and competitions.

 The festival has received worldwide attention and is considered the most prestigious of all hula contests.
</item 7>

<item 8>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 8>

<item 9>
Kalākaua
Kalākaua (November 16, 1836 – January 20, 1891), born David Laʻ amea Kamananakapu Mahinulani Naloiaehuokalani Lumialani Kalākaua and sometimes called The Merrie
Monarch, was the last king (and second to last monarch) of the Kingdom of Hawaiʻ i.

 He reigned from February 12, 1874 until his death in San Francisco, California, on January 20, 1891.

 Kalākaua had a convivial personality and enjoyed entertaining guests with his singing and ukulele playing.

 At his coronation and his birthday jubilee, the hula that had been banned from public in the kingdom became a celebration of Hawaiian culture.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=c197edfab128c03d ===

=== ITEM id=cf57053e016d6424 ===
Gold sentences and facts:
Sentence 0: Dorian Gray is the main character of the novel "The Picture of Dorian Gray" (1890) by Oscar Wilde.
  Fact 0.0: Dorian Gray is the main character of the novel The Picture of Dorian Gray.
  Fact 0.1: The Picture of Dorian Gray was published in 1890.
  Fact 0.2: The Picture of Dorian Gray was written by Oscar Wilde.
Sentence 1: The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".
  Fact 1.0: The Picture of Dorian Gray is a philosophical novel.
  Fact 1.1: The Picture of Dorian Gray was written by Oscar Wilde.
  Fact 1.2: The Picture of Dorian Gray was first published complete.
  Fact 1.3: The Picture of Dorian Gray was first published complete in the July 1890 issue of Lippincott's Monthly Magazine.
Sentence 2:  The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.
  Fact 2.0: The editor of Lippincott's Monthly Magazine feared the story was indecent.
  Fact 2.1: The editor of Lippincott's Monthly Magazine, without Wilde's knowledge, deleted roughly five hundred words before publication.

Retrieved items:
<item 1>
The Picture of Dorian Gray (1945 film)
The Picture of Dorian Gray is a 1945 American horror-drama film based on Oscar Wilde's 1890 novel of the same name.

 Released in March 1945 by Metro-Goldwyn-Mayer, the film is directed by Albert Lewin and stars George Sanders as Lord Henry Wotton and Hurd Hatfield as Dorian
Gray.

 Shot primarily in black-and-white, the film features four inserts in 3-strip Technicolor of Dorian's portrait as a special effect (the first two of his
portrait's original state, and the second two after a major period of degeneration).
</item 1>

<item 2>
The Picture of Dorian Gray (1910 film)
The Picture of Dorian Gray (1910), a.k.a. Dorian Grays Portræt, is a Danish silent film based on the novel "The Picture of Dorian Gray" by Oscar Wilde.
</item 2>

<item 3>
A Portrait of Dorian Gray
A Portrait of Dorian Gray (2005) is the fashion designer and photographer Karl Lagerfeld's rendition of Oscar Wilde's novel "The Picture of Dorian Gray" (1891)
in photography.

 The models, Larry Scott and Eva Herzigová, star as Mr. and Mrs. Dorian Gray.
</item 3>

<item 4>
The Picture of Dorian Gray (1917 film)
The Picture of Dorian Gray (German: Das Bildnis des Dorian Gray) is a 1917 German silent fantasy film directed by Richard Oswald and starring Bernd Aldor, Ernst
Pittschau, and Ernst Ludwig.

 The film is based on the novel "The Picture of Dorian Gray" by Oscar Wilde.
</item 4>

<item 5>
Dorian Gray (1970 film)
Dorian Gray (Italian: "Il dio chiamato Dorian") aka "The Sins of Dorian Gray" is a 1970 movie adaptation of Oscar Wilde's novel "The Picture of Dorian Gray"
starring Helmut Berger.

 The Italian title translates as "A God Called Dorian".
</item 5>

<item 6>
The Picture of Dorian Gray
The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".

 The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.

 Despite that censorship, "The Picture of Dorian Gray" offended the moral sensibilities of British book reviewers, some of whom said that Oscar Wilde merited
prosecution for violating the laws guarding the public morality.

 In response, Wilde aggressively defended his novel and art in correspondence with the British press, although he personally made excisions of some of the most
controversial material when revising and lengthening the story for book publication the following year.
</item 6>

<item 7>
Dorian Gray (disambiguation)
Dorian Gray is the main character of the novel "The Picture of Dorian Gray" (1890) by Oscar Wilde.
</item 7>

<item 8>
Dorian Gray (character)
Dorian Gray is the main character in Oscar Wilde's novel The Picture of Dorian Gray.

 Dorian Gray is a handsome, susceptible, wealthy young Victorian gentleman.

 He falls in love with and devotes himself to Sibyl Vane.

 Dorian has two friends, Basil and Henry.

 Basil is an artist who draws a portrait of Dorian that emphasizes his youthful, noble, and natural beauty.

 Henry is a friend who has a cynical nature and who shares various thoughts with Dorian.
</item 8>

<item 9>
El retrato de Dorian Gray
The Picture of Dorian Gray ("El retrato de Dorian Gray"), is a Mexican telenovela, based on "The Picture of Dorian Gray", the novel written by Oscar Wilde.

 The main character is the handsome young man called Dorian Gray (played by late Enrique Álvarez Félix).

 This telenovela was made in 1969.
</item 9>

<item 10>
Graph query row:
entity: The Picture of Dorian Gray
relationship: CREATED_BY
related: Oscar Wilde
kind: philosophical novel
genre: fantasy
</item 10>
=== END ITEM id=cf57053e016d6424 ===

=== ITEM id=dd6518bb19548edb ===
Gold sentences and facts:
Sentence 0: Dieffenbachia is a genus of tropical flowering plants in the family Araceae.
  Fact 0.0: Dieffenbachia is a genus of flowering plants.
  Fact 0.1: Dieffenbachia is a genus of tropical flowering plants.
  Fact 0.2: Dieffenbachia is in the family Araceae.
Sentence 1: Carlina is a genus of flowering plants in the aster family, Asteraceae.
  Fact 1.0: Carlina is a genus of flowering plants.
  Fact 1.1: Carlina is in the aster family.
  Fact 1.2: Carlina is in the family Asteraceae.

Retrieved items:
<item 1>
Dictyosperma
Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).

 The sole species, Dictyosperma album, is widely cultivated in the tropics but has been farmed to near extinction in its native habitat.

 It is commonly called princess palm or hurricane palm, the latter owing to its ability to withstand strong winds by easily shedding leaves.

 It is closely related to, and resembles, palms in the "Archontophoenix" genus.

 The genus is named from two Greek words meaning "net" and "seed" and the epithet is Latin for "white", the common color of the crownshaft at the top of the
trunk.
</item 1>

<item 2>
Carlina
Carlina is a genus of flowering plants in the aster family, Asteraceae.

 It is distributed from Madeira and the Canary Islands across Europe and northern Africa to Siberia and northwestern China.
</item 2>

<item 3>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 3>

<item 4>
Conophytum
Conophytum is a genus of South African and Namibian succulent plants that belong to the Aizoaceae family.

 The name is derived from the Latin “conus” (cone) and Greek “phytum” (plant).

 The plants are also known as knopies (buttons), waterblasies (water blisters), sphaeroids, conos, cone plants, dumplings, or button plants.
</item 4>

<item 5>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 5>

<item 6>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
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
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 8>

<item 9>
Coronilla
The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
</item 9>

<item 10>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 10>
=== END ITEM id=dd6518bb19548edb ===

=== ITEM id=75af4d84b171388e ===
Gold sentences and facts:
Sentence 0: Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the Himalayas, and Indonesia) and the Americas.
  Fact 0.0: Hydrangea is a genus of flowering plants.
  Fact 0.1: Hydrangea's common names are hydrangea or hortensia.
  Fact 0.2: The Hydrangea genus has 70 to 75 species.
  Fact 0.3: Hydrangea plants are native to southern and eastern Asia.
  Fact 0.4: Hydrangea plants are native to China, Japan, Korea, the Himalayas, and Indonesia.
  Fact 0.5: Hydrangea plants are native to the Americas.
Sentence 1:  It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").
  Fact 1.0: The subject of this sentence contains 13 to 18 species.
  Fact 1.1: The subject of this sentence is best known for a single species.
  Fact 1.2: That single species is Jequirity.
  Fact 1.3: Jequirity is also called A. precatorius.

Retrieved items:
<item 1>
Hydrangea paniculata
Hydrangea paniculata, the panicled hydrangea, is a species of flowering plant in the family Hydrangeaceae native to southern and eastern China, Korea, Japan and
Russia (Sakhalin).

 It was first formally described by Philipp Franz von Siebold in 1829.

 It is a deciduous shrub or small tree, 1 - tall by 2.5 m broad, growing in sparse forests or thickets in valleys or on mountain slopes.
</item 1>

<item 2>
Hydrangea macrophylla
Hydrangea macrophylla is a species of flowering plant in the family Hydrangeaceae, native to Japan.

 It is a deciduous shrub growing to 2 m tall by 2.5 m broad with large heads of pink or blue flowers in summer and autumn.

 Common names include bigleaf hydrangea, French hydrangea, lacecap hydrangea, mophead hydrangea, penny mac and hortensia.

 It is widely cultivated in many parts of the world in many climates.

 It is not to be confused with "H. aspera" 'Macrophylla'.
</item 2>

<item 3>
Hydrangea heteromalla
Hydrangea heteromalla is a species of "Hydrangea" flower, native to the Himalayas and China.

 Average height is 10'-15' with inflorescences that are 5"-8" across and bloom May-June,full flowering only occurs in full sun.Is a very adaptable plant that is
hardy to USDA zones (4)5-7.

 This plant is seldom seen in American gardens although there is nice specimen at the Hillier Arboretum.
</item 3>

<item 4>
Hydrangea arborescens
Hydrangea arborescens, commonly known as smooth hydrangea, wild hydrangea, or sevenbark, is a species of flowering plant in the family Hydrangeaceae.

 It is a small- to medium-sized, deciduous shrub up to 3 m tall that is native to the eastern United States.
</item 4>

<item 5>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 5>

<item 6>
Abrus canescens
Abrus canescens is a species of flowering plant belonging to the legume family, native to Africa.

 It is considered almost extinct.
</item 6>

<item 7>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 7>

<item 8>
Hydrangea serrata
Hydrangea serrata is a species of flowering plant in the family Hydrangeaceae, native to mountainous regions of Korea and Japan.

 Common names include mountain hydrangea and tea of heaven.

 Growing to 1.2 m tall and broad, it is a deciduous shrub with oval leaves and panicles of blue and pink flowers in summer and autumn (fall).

 It is widely cultivated as an attractive ornamental shrub throughout the world in areas with suitable climate and soil.
</item 8>

<item 9>
Abrus
Abrus is a genus of flowering plants in the pea family, Fabaceae and the only genus found in the tribe Abreae.

 It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").

 The highly toxic seeds of that species are used to make jewellery.
</item 9>

<item 10>
Hydrangea quercifolia
Hydrangea quercifolia, commonly known by its translation oakleaf hydrangea or oak-leaved hydrangea, is a species of flowering plant native to the Southeastern
United States, in woodland habitats from North Carolina west to Tennessee, and south to Florida and Louisiana.

 A deciduous shrub with white showy flower heads, it is a commonly grown garden plant.

 Numerous cultivars are available commercially.
</item 10>
=== END ITEM id=75af4d84b171388e ===
