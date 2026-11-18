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

=== ITEM id=f11266030b5e374b ===
Gold sentences and facts:
Sentence 0: Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.
  Fact 0.0: Eric Arthur Blair was born on 25 June 1903.
  Fact 0.1: Eric Arthur Blair died on 21 January 1950.
  Fact 0.2: Eric Arthur Blair's pen name was George Orwell.
  Fact 0.3: Eric Arthur Blair is better known as George Orwell.
  Fact 0.4: Eric Arthur Blair was an English novelist.
  Fact 0.5: Eric Arthur Blair was an English essayist.
  Fact 0.6: Eric Arthur Blair was an English journalist.
  Fact 0.7: Eric Arthur Blair was an English critic.
Sentence 1: Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.
  Fact 1.0: Nelson Richard DeMille was born on August 23, 1943.
  Fact 1.1: Nelson Richard DeMille is an American author.
  Fact 1.2: Nelson Richard DeMille writes action adventure novels.
  Fact 1.3: Nelson Richard DeMille writes suspense novels.
Sentence 2:  DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
  Fact 2.0: Nelson Richard DeMille has written under the pen name Jack Cannon.
  Fact 2.1: Nelson Richard DeMille has written under the pen name Kurt Ladner.
  Fact 2.2: Nelson Richard DeMille has written under the pen name Ellen Kay.
  Fact 2.3: Nelson Richard DeMille has written under the pen name Brad Matthews.

Retrieved items:
<item 1>
George Orwell: A Life in Pictures
George Orwell: A Life in Pictures is a 2003 BBC Television docudrama telling the life story of the British author George Orwell.

 Chris Langham plays the part of Orwell.

 No surviving sound recordings or video of the real George Orwell have been found.
</item 1>

<item 2>
The Lion (2010 novel)
The Lion is a 2010 novel by American author Nelson DeMille.

 It is the fifth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "Wild Fire".

 The Lion is followed by DeMille’s 2012 novel, "The Panther".
</item 2>

<item 3>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 3>

<item 4>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 4>

<item 5>
The Lion's Game
The Lion's Game is a 2000 novel by American author Nelson DeMille.

 It is the second of DeMille's novels to feature the detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New
York City.

 The 2000 novel The Lion’s Game is the sequel to Plum Island and is followed by the 2004 book Night Fall.

 The book also briefly mentions events from other DeMille novels like "The Charm School" and "The Gold Coast", that aren't strictly part of the Corey-universe.
</item 5>

<item 6>
The Panther (novel)
The Panther is a 2012 novel by American author Nelson DeMille.

 It is the sixth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "The Lion".

 The Panther is followed by DeMille’s 2015 novel, "Radiant Angel".

 Also featured in this novel is DeMille’s other fictional character, Paul Brenner, who appears in The General's Daughter and Up Country.
</item 6>

<item 7>
George Orwell
Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.

 His work is marked by lucid prose, awareness of social injustice, opposition to totalitarianism, and outspoken support of democratic socialism.
</item 7>

<item 8>
Wild Fire (novel)
Wild Fire is a 2006 novel by American author, Nelson DeMille.

 It is the fourth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to Night Fall and takes place approximately one year later.

 Wild Fire is followed by DeMille’s 2010 novel, The Lion.
</item 8>

<item 9>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 9>

<item 10>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 10>
=== END ITEM id=f11266030b5e374b ===

=== ITEM id=c4324ed77e09531a ===
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
Weight Wins
Weight Wins is a UK weight loss business which has developed and commercialised the concept of personal contracts for weight loss.

 The company has attracted attention from UK and international media for its ‘Pounds for Pounds’ weight loss programme and for operating the first trial by the
UK’s National Health Service (NHS) of a weight loss incentive programme.
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
Erno S. Daniel
Erno S. Daniel M.D. Ph.D (Born December 15, 1946, Died February 21, 2015) is an internal medicine specialist and author.

 He is of Hungarian descent.

 He has practiced at the Sansum Clinic in Santa Barbara (formerly known as the Santa Barbara Medical Foundation Clinic) since 1978.

 Since that time, he has participated in teaching activities at Santa Barbara's Cottage Hospital, whose internal medicine residency training program is
affiliated with University of Southern California School of Medicine.

 Because of his activities, for a period of time Dr. Daniel held an appointment as Associate Clinical Professor of Medicine at USC.

 He also participated as a lecturer for the Introduction to Medicine course offered by the University of California at Santa Barbara.
</item 6>

<item 7>
Eat to Live
Eat to Live: The Amazing Nutrient-Rich Program for Fast and Sustained Weight Loss is a book written in 2003 by Dr. Joel Fuhrman, which he revised in 2011.

 The book offers a formula: health = nutrients / calories.
</item 7>

<item 8>
Management of obesity
Management of obesity can include lifestyle changes, medications, or surgery.

 The main treatment for obesity consists of dieting and physical exercise.

 Diet programs may produce weight loss over the short term, but maintaining this weight loss is frequently difficult and often requires making exercise and a
lower calorie diet a permanent part of an individual's lifestyle.

 Success rates of long-term weight loss maintenance with lifestyle changes are low, ranging from 2 to 20%.

 Dietary and lifestyle changes are effective in limiting excessive weight gain in pregnancy and improve outcomes for both the mother and the child.

 The National Institutes of Health recommend a weight loss goal of 5% to 10% of the person's current weight over six months.
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
=== END ITEM id=c4324ed77e09531a ===

=== ITEM id=6665a28358da5795 ===
Gold sentences and facts:
Sentence 0: Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.
  Fact 0.0: Cassiope is a genus of 9 to 12 small shrubby species.
  Fact 0.1: Cassiope is in the family Ericaceae.
Sentence 1: Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to Japan and the Philippines), and Central America and also Europe.
  Fact 1.0: Deutzia is a genus of about 60 species of flowering plants.
  Fact 1.1: Deutzia is in the family Hydrangeaceae.
  Fact 1.2: Deutzia is native to eastern Asia.
  Fact 1.3: Deutzia is native to central Asia.
  Fact 1.4: Deutzia is native from the Himalayas east to Japan and the Philippines.
  Fact 1.5: Deutzia is native to Central America.
  Fact 1.6: Deutzia is also native to Europe.

Retrieved items:
<item 1>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 1>

<item 2>
Cassiope lycopodioides
Cassiope lycopodioides, Haida Gwaii mountain-heather or clubmoss mountain heather, is a plant species native to southern Alaska, British Columbia, and the US
State of Washington.

 It is found on rocky slopes in arctic and alpine tundra at elevations up to 2000 m.

 In Washington, it is reported only from King County.

 The specific epithet ""lycopodioides"" refers to the plant's superficial resemblance to some species of clubmoss "(Lycopodium" sensu lato).
</item 2>

<item 3>
Aizoaceae
Aizoaceae (the fig-marigold family) is a large family of dicotyledonous flowering plants containing 135 genera and about 1900 species.

 They are commonly known as stone plants or carpet weeds.

 They are often called vygies in South Africa and New Zealand.

 Species that resemble stones or pebbles are sometimes called mesembs.

 Several species are known as ice plants because of the glistening globular bladder cells covering their stems, fruit and leaves, "... they sparkle like ice
crystals."
</item 3>

<item 4>
Cassiope mertensiana
Cassiope mertensiana is a species of flowering plant known by the common names western moss heather and white mountain heather.
</item 4>

<item 5>
Deutzia silvestrii
Deutzia silvestrii is a plant species in the family Hydrangeaceae.
</item 5>

<item 6>
Deutzia ningpoensis
Deutzia ningpoensis is a shrub in the family Hydrangeaceae.

 The species is endemic to China.

 It grows to between 1 and 2.5 metres high and produces panicles of white flowers from May to July in its native range.
</item 6>

<item 7>
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 7>

<item 8>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 8>

<item 9>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 9>

<item 10>
Cassiope
Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.

 They are native to the Arctic and north temperate montane regions.

 The genus is named after Cassiopeia of Greek mythology.

 Common names, shared with several other similar related genera, include heather and heath.

 They have scale-like leaves lying against the stems, and produce solitary bell-shaped flowers in late spring.

 Though hardy, flowers can be damaged by late frosts.
</item 10>
=== END ITEM id=6665a28358da5795 ===

=== ITEM id=89afd419c6a3a6a6 ===
Gold sentences and facts:
Sentence 0:  In 2010, he began providing vocals and guitar in the pop rock band Ashland HIGH.
  Fact 0.0: In 2010, the person referred to as "he" began providing vocals in the pop rock band Ashland HIGH.
  Fact 0.1: In 2010, the person referred to as "he" began providing guitar in the pop rock band Ashland HIGH.
Sentence 1: Jizzy Pearl (born James Wilkinson) is an American vocalist that was born on March 17, 1958, in Chicago, Illinois.
  Fact 1.0: Jizzy Pearl was born James Wilkinson.
  Fact 1.1: Jizzy Pearl is an American vocalist.
  Fact 1.2: Jizzy Pearl was born on March 17, 1958.
  Fact 1.3: Jizzy Pearl was born in Chicago, Illinois.

Retrieved items:
<item 1>
Jizzy Pearl
Jizzy Pearl (born James Wilkinson) is an American vocalist that was born on March 17, 1958, in Chicago, Illinois.

 He's had notable performances in several heavy metal and hard rock groups.

 He first fronted the band Data Clan, which eventually became Love/Hate.

 Pearl has also sung for L.A. Guns, Ratt, Adler's Appetite, and other, lesser known acts.
</item 1>

<item 2>
Brother Clyde (album)
Brother Clyde is the debut studio album from alternative rock group Brother Clyde.

 It is the first side-project from country music singer Billy Ray Cyrus.

 The album features duets with Cyrus' son Trace Cyrus as well as Dolly Parton, who Cyrus said "wanted to rock."

 It was released on August 10, 2010.

 Its first single, "Lately", which features rapper King Phaze, was issued to digital retailers on June 29, 2010, and a music video, which was directed by Cyrus,
was released for the song in July 2010.
</item 2>

<item 3>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 3>

<item 4>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 4>

<item 5>
Trace Cyrus
Trace Dempsey Cyrus (born Neil Timothy Helson; February 24, 1989) is an American musician.

 The son of country singer Billy Ray Cyrus and brother of recording artist Miley Cyrus, he is the backing vocalist and bassist of the band Metro Station.

 He left the band in 2010 but returned in 2014.

 In 2010, he began providing vocals and guitar in the pop rock band Ashland HIGH.

 He also owns the clothing company From Backseats to Bedrooms.
</item 5>

<item 6>
Shrinking Violet (album)
Shrinking Violet is the sixth album by glam metal band L.A. Guns, first released on June 1, 1999, through Perris Records, and is the only L.A. Guns album with
singer Jizzy Pearl.

 The album was reissued, with bonus tracks and new artwork, on May 24, 2010, through Favored Nations.

 The album was produced by former Guns N' Roses guitarist Gilby Clarke who also provided additional guitars on the track "Dreamtime".
</item 6>

<item 7>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 7>

<item 8>
Quiet Riot 10
Quiet Riot 10 (also alternatively known as just 10, or Quiet Riot Number 10) is the twelfth studio album by the heavy metal band Quiet Riot, which was released
on June 27, 2014.

 It is their first studio album since 1988's "QR" not to feature longtime and founding vocalist Kevin DuBrow in any newly recorded material, due to his death in
November 2007.

 It is also the band's first album since reuniting in 2010.

 Although a studio album, the final four tracks on "Quiet Riot 10" are live performances taken from some of the band's final shows with DuBrow in 2007.

 Love/Hate vocalist Jizzy Pearl joined the band in November 2013 and performs lead vocals on the six studio tracks.
</item 8>

<item 9>
Zachary Throne
As a singer, Zachary has sung on many TV and radio jingles.

 He provided the singing voice for the character, "Mark Winkle" on the television series, "California Dreams" as well as the singing voice for the character,
"Greg Brady" in the films, "The Brady Bunch Movie" and "A Very Brady Sequel".

 Currently, he resides in Las Vegas, Nevada where he performs in many shows.

 From 2012-2014, he was the lead singer/lead guitar player/bass player for the Sin City Sinners, a group that also featured former Faster Pussycat guitarist and
co-founder Brent Muscat as well as Slash bass player, Todd Kerns.

 .

 With the group, he recorded two albums, "DIVEBAR Days Revisited" and "A Sinners Christmas 2", both released in 2013.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=89afd419c6a3a6a6 ===

=== ITEM id=94b40da34f7e661e ===
Gold sentences and facts:
Sentence 0: Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.
  Fact 0.0: Warrel Dane was born on March 7, 1969.
  Fact 0.1: Warrel Dane is the lead singer of the band Sanctuary.
  Fact 0.2: Sanctuary is an American power metal band.
Sentence 1: Roy Sætre Khantatat (born March 12, 1970), is a Norwegian singer-songwriter.
  Fact 1.0: Roy Sætre Khantatat was born on March 12, 1970.
  Fact 1.1: Roy Sætre Khantatat is a Norwegian singer-songwriter.

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
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 2>

<item 3>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 3>

<item 4>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 4>

<item 5>
John Warhola
John Warhola (May 31, 1925 – December 24, 2010) played a pivotal role in maintaining the legacy of his younger brother, pop artist Andy Warhol, assigned
responsibility by their father on his deathbed to ensure that Andy attended college and serving as a trustee of the Andy Warhol Foundation for the Visual Arts
after his brother's death in 1987.

 Warhola oversaw the establishment of The Andy Warhol Museum in Pittsburgh and the Andy Warhol Museum of Modern Art in Medzilaborce, Slovakia.
</item 5>

<item 6>
Roy Khan
Roy Sætre Khantatat (born March 12, 1970), is a Norwegian singer-songwriter.

 Commonly known as Roy Khan or simply Khan, he was the lead vocalist for the symphonic power metal band Kamelot from 1997 to his departure in 2011.

 He co-wrote most of Kamelot's songs with the band's guitarist and founder Thomas Youngblood during his tenure.

 Prior to joining Kamelot, Khan was lead singer for the Norwegian progressive metal band Conception from 1991 until 1997 when they disbanded.
</item 6>

<item 7>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 7>

<item 8>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 8>

<item 9>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 9>

<item 10>
Sanctuary (band)
Sanctuary is an American heavy metal band founded in Seattle in 1985.

 They split up in 1992, but reformed 18 years later.

 The band consists of Warrel Dane (vocals), Lenny Rutledge (guitar), Nick Cordle (guitar), George Hernandez (bass), and Dave Budbill (drums).

 To date, they have released four studio albums and one live EP.
</item 10>
=== END ITEM id=94b40da34f7e661e ===

=== ITEM id=69079829ef36c263 ===
Gold sentences and facts:
Sentence 0: Enrico Cocozza (6 November 1921 – 27 December 1997), was a Scottish filmmaker who won many film awards during the 1940s and 1950s.
  Fact 0.0: Enrico Cocozza was born on 6 November 1921.
  Fact 0.1: Enrico Cocozza died on 27 December 1997.
  Fact 0.2: Enrico Cocozza was a Scottish filmmaker.
  Fact 0.3: Enrico Cocozza won many film awards during the 1940s and 1950s.
Sentence 1: ", which was nominated for the Academy Award for Best Foreign Language Film.
  Fact 1.0: A film was nominated for the Academy Award for Best Foreign Language Film.
  Fact 1.1: The film whose title is quoted in this fragment was nominated for the Academy Award for Best Foreign Language Film.

Retrieved items:
<item 1>
Academy Award for Best Film Editing
The Academy Award for Best Film Editing is one of the annual awards of the Academy of Motion Picture Arts and Sciences (AMPAS).

 Nominations for this award are closely correlated with the Academy Award for Best Picture.

 For 33 consecutive years, 1981 to 2013, every Best Picture winner had also been nominated for the Film Editing Oscar, and about two thirds of the Best Picture
winners have also won for Film Editing.

 Only the principal, "above the line" editor(s) as listed in the film's credits are named on the award; additional editors, supervising editors, etc. are not
currently eligible.

 The nominations for this Academy Award are determined by a ballot of the voting members of the Editing Branch of the Academy; there were 220 members of the
Editing Branch in 2012.

 The members may vote for up to five of the eligible films in the order of their preference; the five films with the largest vote totals are selected as
nominees.

 The Academy Award itself is selected from the nominated films by a subsequent ballot of all active and life members of the Academy.
</item 1>

<item 2>
Shafiq Syed
Shafiq Syed (born 1976) is a former Indian child actor turned auto rickshaw driver.

 At the age of twelve, he starred in a lead role in the Mira Nair's Academy Award-nominated film "Salaam Bombay!

" (1988), for which he also received the 1989 National Film Award for Best Child Artist at the 36th National Film Awards.

 He drives an auto rickshaw in Bangalore and also works as an assistant in television production units.
</item 2>

<item 3>
Enrico Cocozza
Enrico Cocozza (6 November 1921 – 27 December 1997), was a Scottish filmmaker who won many film awards during the 1940s and 1950s.

 His often surreal films were mainly filmed in and around the town of Wishaw in Scotland, where his family owned the popular Belhaven Cafe.

 These include "Chick's Day" (1950), a prize winner at the 1951 Scottish Amateur Film Festival, "The Living Ghost" (1957), and "Glasgow's Docklands" (1959).

 Illness later forced him to give up making films and he spent most of his working life teaching at the University of Strathclyde in Glasgow.

 In 2001 he was the subject of the Channel 4 documentary "Artery: the Story of Enrico Cocozza".
</item 3>

<item 4>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 4>

<item 5>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 5>

<item 6>
Salaam Bombay!
Salaam Bombay!

 is a 1988 Hindi film directed by Mira Nair, and screenwritten by her longtime creative collaborator, Sooni Taraporevala.

 The film chronicles the day-to-day life of children living on the streets of Bombay, India's biggest city.

 It won the National Film Award for Best Feature Film in Hindi, the National Board of Review Award for Top Foreign Film, the Golden Camera and Audience Awards
at the Cannes Film Festival, and three awards at the Montréal World Film Festival.

 The film was India's second film submission to be nominated for the Academy Award for Best Foreign Language Film.

 The film was among the list of "The Best 1,000 Movies Ever Made" by "The New York Times".
</item 6>

<item 7>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 7>

<item 8>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 8>

<item 9>
Shamal Sabri
He has proven himself over and over as an important player in Kurdistan's developing film industry and eventually earned the position of Artistic Director of
Duhok IFF in 2016.

 Shamal is the head of submission committee for films representing Iraq in the Asia Pacific Screen Awards.
</item 9>

<item 10>
Maya Deren Award
The American Film Institute Award for Independent Film and Video Artists, subtitled and generally known as the Maya Deren Award, was an award presented to
filmmakers and video artists by the American Film Institute to honor independent filmmaking.

 Named for the avant-garde experimental film artist Maya Deren, it was given from 1986 through 1996.
</item 10>
=== END ITEM id=69079829ef36c263 ===

=== ITEM id=0fc4996087ba473c ===
Gold sentences and facts:
Sentence 0: The University of Arizona (also referred to as U of A, UA, or Arizona) is a public research university in Tucson, Arizona, United States.
  Fact 0.0: The University of Arizona is a public research university.
  Fact 0.1: The University of Arizona is in Tucson, Arizona, United States.
  Fact 0.2: The University of Arizona is also referred to as U of A.
  Fact 0.3: The University of Arizona is also referred to as UA.
  Fact 0.4: The University of Arizona is also referred to as Arizona.
Sentence 1: The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.
  Fact 1.0: The Presbyterian University College, Ghana is a partially private and public university.
  Fact 1.1: The Presbyterian University College, Ghana is located at Abetifi-Kwahu.
  Fact 1.2: Abetifi-Kwahu is in the Eastern Region of Ghana.
Sentence 2:  It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.
  Fact 2.0: The Presbyterian University College, Ghana is one of the new universities in Ghana.
  Fact 2.1: The Presbyterian University College, Ghana was granted accreditation by the National Accreditation Board.

Retrieved items:
<item 1>
Presbyterian University College
The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.

 It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.

 It was established by the Presbyterian Church of Ghana on 23 November 2003 and inaugurated on 27 March 2004 by the president of Ghana, John Agyekum Kufuor.
</item 1>

<item 2>
Prempeh College
Prempeh College is an elite public secondary school for boys located in Kumasi, the capital city of the Ashanti Region, Ghana.

 The school was founded in 1949 by the Asanteman traditional authority, the British Colonial Government, the Methodist Church of Ghana and the Presbyterian
church of Ghana.

 The School is named after King of Ashanti, (Asantehene) Sir Osei Tutu Agyeman Prempeh II, who donated the land on which the school was built.

 It is the oldest government-assisted secondary school for boys in the Ashanti and was modeled on Eton College in England.

.

 The school topped matriculation at the Kwame Nkrumah University of Science and Technology in 2004 with 441 students admitted and in 2012, with 296 students
from the college admitted, and is considered to be one of the best secondary schools in Ghana.

 The School has won the National robotics championships a record three times between 2013 and 2016 and currently is the only school from Africa to have won the
International robofest World Championships.
</item 2>

<item 3>
Albanian University in Berat
The University of Berat is a private institution, independent and secular Higher Education, which operates the state fee.

 Licensed by the Council of Ministers, no. 697 dated 18.06.2009, he opened the doors in the academic year 2009- 2010, offering programs in two faculties:
Faculty of Social Sciences Programs in Science Teaching Legal, Political Science and Psychology General Administration and the Faculty of Economics with
Learning programs in Banking and Finance-Business Administration.

 During the first academic year, about 1,000 students attended studies.

 A year later, this new institution offering undergraduate degrees in study programs: Preschool Teacher Education, Primary Education Teacher, English Language,
Nursing, Engineering Computer and Information Technology, thus finalizing the 2010-2011 academic year with the opening of two new faculties: Faculty of Medical
Sciences and the Faculty of Architecture and Inxhinierive.

 Academic Year 2010-2011 finds the University Berat with about 2600 students in the first cycle of study.

 Since the first year of its inception, the institution pointed at the center of academic life, the promotion of intellectual product and melting learning and
research.
</item 3>

<item 4>
Aaron McCreary
Aaron Monroe "Mac" McCreary (September 15, 1892 – ?)

 was an American football, basketball, and baseball player, coach, and college athletics administrator.

 He served as the head football coach at Tempe State Teachers College, known at Arizona State Teachers College by 1929 and now called Arizona State University,
compiling a career college football record of 25–17–4.

 McCreary was also the head basketball coach at Tempe/Arizona State Teachers from 1923 to 1930 and at Arizona State Teacher's College of Flagstaff, now Northern
Arizona University, amassing a career college basketball record of 140–149.

 In addition, he coached baseball at Tempe/Arizona State Teachers (1924–1926, 1928, 1930–1931) and at Arizona State Teacher's Flagstaff in 1959, tallying a
career college baseball mark of 22–56–1.
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
Evangelical Presbyterian University College
The Evangelical Presbyterian University College was established in 2008 by the Evangelical Presbyterian Church, Ghana.
</item 6>

<item 7>
Syracuse Triad
The Syracuse Triad is the name given to the three women's sororities founded at Syracuse University.

 Alpha Phi was founded first in 1872 by 10 of the original 20 women admitted into Syracuse University.

 Gamma Phi Beta came along two years later in 1874 and with it came the term "sorority," which was coined at the time of its founding.

 (Prior to that, women's Greek-letter organizations used the term "women's fraternity," since no more appropriate term existed.)

 Alpha Gamma Delta completed the triad in 1904.
</item 7>

<item 8>
Lincoln University College, Malaysia
Lincoln University College, Petaling Jaya, Malaysia, was founded in the year in 2002 as Lincoln College and in 2011 Lincoln College was upgraded to Lincoln
University College.

 Lincoln University College is a private institution that provides higher education, approved by the Malaysian Qualifications Agency (MQA, National
Accreditation Board) and the Ministry of Higher Education.

 Situated in the cosmopolitan town of Petaling Jaya, the university college is close to the capital city, Kuala Lumpur.
</item 8>

<item 9>
University of Arizona
The University of Arizona (also referred to as U of A, UA, or Arizona) is a public research university in Tucson, Arizona, United States.

 Founded in 1885, the UA was the first university in the Arizona Territory.

 The university operates two medical schools (University of Arizona College of Medicine - Tucson and the University of Arizona College of Medicine - Phoenix)
and is affiliated with the region's only academic medical centers (Banner - University Medical Center Tucson and Banner - University Medical Center Phoenix).

 The university is also home to the James E. Rogers College of Law and numerous other nationally ranked graduate and professional schools.

 During the 2016–2017 academic year, there was a total enrollment of 43,625 students, including 34,072 undergraduates The University of Arizona is governed by
the Arizona Board of Regents.

 The University of Arizona is one of the elected members of the Association of American Universities (an organization of North America's premier research
institutions) and is the only representative from the state of Arizona to this group.
</item 9>

<item 10>
Alpha Gamma
Alpha Gamma (ΑΓ) was an American collegiate fraternity.

 It was founded at Cumberland University in Lebanon, Tennessee in 1867.

 About 21 chapters developed, with prominent chapters at Washington & Jefferson College, Trinity University, Mercersburg College, Southwestern Presbyterian
University, Cumberland University, and the West Virginia University.

 The fraternity's badge consisted of a golden shield with a globe, six stars, and a pennant with "ΑΓ."
</item 10>
=== END ITEM id=0fc4996087ba473c ===

=== ITEM id=604b3ba956014c7b ===
Gold sentences and facts:
Sentence 0:  Related fields of study include Islamic philosophy and Sufism.
  Fact 0.0: Related fields of study include Islamic philosophy.
  Fact 0.1: Related fields of study include Sufism.
Sentence 1: Sufism or Taṣawwuf (Arabic: الْتَّصَوُّف [sect]; personal noun: صُوفِيّ - "ṣūfiyy"/ "ṣūfī", مُتَصَوّف - "mutaṣawwūf"), which is often defined as "Islamic mysticism", "the inward dimension of Islam", or "the phenomenon of mysticism within Islam", is a mystical trend in Islam "characterized ... [by particular] values, ritual practices, doctrines and institutions" which began very early on in Islamic history and which represents "the main manifestation and the most important and central crystallization of" mystical practice in Islam.
  Fact 1.0: Sufism is also called Taṣawwuf.
  Fact 1.1: The Arabic name for Sufism is الْتَّصَوُّف [sect].
  Fact 1.2: The personal noun for a Sufi is صُوفِيّ, transliterated as 'ṣūfiyy' or 'ṣūfī'.
  Fact 1.3: The personal noun mutaṣawwūf is مُتَصَوّف.
  Fact 1.4: Sufism is often defined as 'Islamic mysticism'.
  Fact 1.5: Sufism is often defined as 'the inward dimension of Islam'.
  Fact 1.6: Sufism is often defined as 'the phenomenon of mysticism within Islam'.
  Fact 1.7: Sufism is a mystical trend in Islam.
  Fact 1.8: Sufism is characterized by particular values, ritual practices, doctrines and institutions.
  Fact 1.9: Sufism began very early on in Islamic history.
  Fact 1.10: Sufism represents the main manifestation of mystical practice in Islam.
  Fact 1.11: Sufism represents the most important and central crystallization of mystical practice in Islam.

Retrieved items:
<item 1>
Moinuddin Chishti
1325) and Amir Khusrow (d. 1325).

 As such, Muʿīn al-Dīn Chishtī's legacy rests primarily on his having been "one of the most outstanding figures in the annals of Islamic mysticism."

 Additionally, Muʿīn al-Dīn Chishtī is also notable for having been one of the first major Islamic mystics to formally allow his followers to incorporate the
"use of music" in their devotions, liturgies, and hymns to God, which he did so in order to make the foreign Arab faith more relatable to the indigenous peoples
who had recently entered the religion or whom he sought to convert.
</item 1>

<item 2>
Animality studies
Animality studies is an emerging interdisciplinary academic field focused on the cultural study of animals and animality.

 It can be distinguished from animal studies and critical animal studies by its resistance to animal rights or animal welfare as an explicit justification for
work in this field.

 According to Michael Lundblad, "If animal studies can be seen as work that explores representations of animality and related discourses with an emphasis on
advocacy for nonhuman animals, animality studies becomes work that emphasizes the history of animality in relation to human cultural studies, without an
explicit call for nonhuman advocacy."
</item 2>

<item 3>
Reynold A. Nicholson
Reynold Alleyne Nicholson, or R. A. Nicholson (18 August 1868 – 27 August 1945), was an eminent English orientalist, scholar of both Islamic literature and
Islamic mysticism and widely regarded as one of the greatest Rumi (Mevlana or Mawlana) scholars and translators in the English language.
</item 3>

<item 4>
Ovamir Anjum
Ovamir Anjum is the Imam Khattab Chair of Islamic Studies at the Department of Philosophy, University of Toledo.

 He studies the connections between theology, ethics, politics, and law in classical and medieval Islam, with a subfocus on its comparisons with western
thought.

 Related fields of study include Islamic philosophy and Sufism.
</item 4>

<item 5>
A Hidden Treasure
Hidden Treasure (Arabic: کنزاً مخفیاً‎ ‎ ) is a Hadith Qudsi that has a very prominent role in Islamic mysticism and Islamic philosophy.
</item 5>

<item 6>
Omid Safi
Omid Safi is an American Professor of Asian and Middle Eastern Studies at Duke University, where he is the Director of Duke Islamic Studies Center, and a
columnist for On Being.

 Dr. Safi specializes in Islamic mysticism (Sufism), contemporary Islamic thought and medieval Islamic history.

 He has served on the board of the Pluralism project at Harvard University and is the co-chair of the steering committee for the Study of Islam and the Islamic
Mysticism Group at the American Academy of Religion.

 Before joining Duke University, Dr. Safi was a professor at the University of North Carolina Chapel Hill.
</item 6>

<item 7>
Sufism
Sufism or Taṣawwuf (Arabic: الْتَّصَوُّف [sect]; personal noun: صُوفِيّ - "ṣūfiyy"/ "ṣūfī", مُتَصَوّف - "mutaṣawwūf"), which is often defined as "Islamic
mysticism", "the inward dimension of Islam", or "the phenomenon of mysticism within Islam", is a mystical trend in Islam "characterized ... [by particular]
values, ritual practices, doctrines and institutions" which began very early on in Islamic history and which represents "the main manifestation and the most
important and central crystallization of" mystical practice in Islam.

 Although the overwhelming majority of Sufis, both pre-modern and modern, have been adherents of Sunni Islam, there nevertheless also developed certain strands
of Sufi practice within the ambit of Shia Islam during the late medieval period.
</item 7>

<item 8>
Irfan
In Islam, ‘Irfaan (Arabic/Persian/Urdu: عرفان ; Turkish: "İrfan" ), also spelt "Irfaan" and "Erfan", literally ‘knowledge, awareness, wisdom’, is gnosis.

 Islamic mysticism can be considered as a vast range that engulfs theoretical and practical and conventional mysticism and has been intertwined with sufism and
in some cases they are assumed identical.

 however Islamic mysticism is assumed as one of the Islamic sciences alongside theology and philosophy.

 Islamic’s mysticism is cognition and knowledge that love has been intertwined through it with structure of revelation in Islam.
</item 8>

<item 9>
Madonna Studies
Madonna Studies is a term which has been used to refer to a development of a field in several studios since late 1980s.

 One writer described the "institutionalization of a major subdivision of American media studies into Madonna studies".

 The field related to the study of and interpretation of the work of American pop musician Madonna using an interdisciplinary approach incorporating cultural
studies and media studies.

 A notable compendium of essays titled "The Madonna Connection" was published in 1992.

 Controversy over this field of study stemmed from discussions over the intellectual worth of pursuing academic inquiry into a pop musician, with some arguing
the field was nothing more than pop cultural commentary.
</item 9>

<item 10>
Moinuddin Chishti
Chishtī Muʿīn al-Dīn Ḥasan Sijzī (1142–1236 CE), known more commonly as Muʿīn al-Dīn Chishtī or Moinuddin Chishti, or reverently as a Shaykh Muʿīn al-Dīn or
Hazrat Muʿīn al-Dīn or Khwājā Muʿīn al-Dīn by South Asian Muslims, was a Persian Muslim preacher, ascetic, religious scholar, philosopher, and mystic from
Sistan, who eventually ended up settling in the Indian subcontinent in the early 13th-century, where he promulgated the famous Chishtiyya order of Sunni
mysticism.

 This particular "tariqa" (order) became the dominant Muslim spiritual group in medieval India and many of the most beloved and venerated Indian Sunni saints
were Chishti in their affiliation, including Nizamuddin Awliya (d. 1325) and Amir Khusrow (d. 1325).
</item 10>
=== END ITEM id=604b3ba956014c7b ===

=== ITEM id=fd91c285575c6ef2 ===
Gold sentences and facts:
Sentence 0:  Related fields of study include Islamic philosophy and Sufism.
  Fact 0.0: Related fields of study include Islamic philosophy.
  Fact 0.1: Related fields of study include Sufism.
Sentence 1: Sufism or Taṣawwuf (Arabic: الْتَّصَوُّف [sect]; personal noun: صُوفِيّ - "ṣūfiyy"/ "ṣūfī", مُتَصَوّف - "mutaṣawwūf"), which is often defined as "Islamic mysticism", "the inward dimension of Islam", or "the phenomenon of mysticism within Islam", is a mystical trend in Islam "characterized ... [by particular] values, ritual practices, doctrines and institutions" which began very early on in Islamic history and which represents "the main manifestation and the most important and central crystallization of" mystical practice in Islam.
  Fact 1.0: Sufism is also called Taṣawwuf.
  Fact 1.1: The Arabic name for Sufism is الْتَّصَوُّف [sect].
  Fact 1.2: The personal noun for a Sufi is صُوفِيّ, transliterated as 'ṣūfiyy' or 'ṣūfī'.
  Fact 1.3: The personal noun mutaṣawwūf is مُتَصَوّف.
  Fact 1.4: Sufism is often defined as 'Islamic mysticism'.
  Fact 1.5: Sufism is often defined as 'the inward dimension of Islam'.
  Fact 1.6: Sufism is often defined as 'the phenomenon of mysticism within Islam'.
  Fact 1.7: Sufism is a mystical trend in Islam.
  Fact 1.8: Sufism is characterized by particular values, ritual practices, doctrines and institutions.
  Fact 1.9: Sufism began very early on in Islamic history.
  Fact 1.10: Sufism represents the main manifestation of mystical practice in Islam.
  Fact 1.11: Sufism represents the most important and central crystallization of mystical practice in Islam.

Retrieved items:
<item 1>
Moinuddin Chishti
1325) and Amir Khusrow (d. 1325).

 As such, Muʿīn al-Dīn Chishtī's legacy rests primarily on his having been "one of the most outstanding figures in the annals of Islamic mysticism."

 Additionally, Muʿīn al-Dīn Chishtī is also notable for having been one of the first major Islamic mystics to formally allow his followers to incorporate the
"use of music" in their devotions, liturgies, and hymns to God, which he did so in order to make the foreign Arab faith more relatable to the indigenous peoples
who had recently entered the religion or whom he sought to convert.
</item 1>

<item 2>
Irfan
In Islam, ‘Irfaan (Arabic/Persian/Urdu: عرفان ; Turkish: "İrfan" ), also spelt "Irfaan" and "Erfan", literally ‘knowledge, awareness, wisdom’, is gnosis.

 Islamic mysticism can be considered as a vast range that engulfs theoretical and practical and conventional mysticism and has been intertwined with sufism and
in some cases they are assumed identical.

 however Islamic mysticism is assumed as one of the Islamic sciences alongside theology and philosophy.

 Islamic’s mysticism is cognition and knowledge that love has been intertwined through it with structure of revelation in Islam.
</item 2>

<item 3>
Shia Muslims in the Arab world
Islam is historically divided into two major sects, Sunni and Shia Islam, each with its own sub-sects.

 Large numbers of Shia Arab Muslims live in some Arab countries including Lebanon, Yemen, Iraq, Saudi Arabia, Kuwait, Oman, the UAE, and Qatar.

 Shia Muslims are a numerical majority in Iraq and Bahrain.

 Approximately half of the population in Yemen are Shia Muslims.

 Exactly half of Muslims in Lebanon are Shia Muslims.

 There is also a very large population of Shia Muslims living in the Arab Persian Gulf countries especially in Saudi Arabia.

 Approximately the whole population of East Saudi Arabia, the Eastern Province are Shia Muslims.

 Although government statistics claim that roughly only 20-40% of the Muslim population are Shia Muslims, there has been dispute to the authenticity of this
figure and recent reports and investigations indicate that there is in fact a much larger population of Shia Muslims present, with estimate figures over 45% or
even making the majority of Muslim population.

 Saudi Arabia follows a strict recently established sect of Islam, Wahhabism, there is little freedom of religion between the different sects even whilst all of
the population are Muslims.
</item 3>

<item 4>
Omid Safi
Omid Safi is an American Professor of Asian and Middle Eastern Studies at Duke University, where he is the Director of Duke Islamic Studies Center, and a
columnist for On Being.

 Dr. Safi specializes in Islamic mysticism (Sufism), contemporary Islamic thought and medieval Islamic history.

 He has served on the board of the Pluralism project at Harvard University and is the co-chair of the steering committee for the Study of Islam and the Islamic
Mysticism Group at the American Academy of Religion.

 Before joining Duke University, Dr. Safi was a professor at the University of North Carolina Chapel Hill.
</item 4>

<item 5>
Sufism
Sufism or Taṣawwuf (Arabic: الْتَّصَوُّف [sect]; personal noun: صُوفِيّ - "ṣūfiyy"/ "ṣūfī", مُتَصَوّف - "mutaṣawwūf"), which is often defined as "Islamic
mysticism", "the inward dimension of Islam", or "the phenomenon of mysticism within Islam", is a mystical trend in Islam "characterized ... [by particular]
values, ritual practices, doctrines and institutions" which began very early on in Islamic history and which represents "the main manifestation and the most
important and central crystallization of" mystical practice in Islam.

 Although the overwhelming majority of Sufis, both pre-modern and modern, have been adherents of Sunni Islam, there nevertheless also developed certain strands
of Sufi practice within the ambit of Shia Islam during the late medieval period.
</item 5>

<item 6>
Abdal
Abdāl ("lit": substitutes, but which can also mean "generous" ["karīm"] and "noble" ["sharīf"]) is a term used in Islamic metaphysics and Islamic mysticism,
both Sunni and Shiite, to refer to a particularly important group of God's saints.

 In the tradition of Sunni Islam in particular, the concept attained an especially important position in the writings of the Sunni mystics and theologians,
whence it appears in the works of Sunni authorities as diverse as Abu Talib al-Makki (d. 956), Ali Hujwiri (d. 1072), Ibn Asakir (d. 1076), Khwaja Abdullah
Ansari (d. 1088), and Ibn Khaldun (d. 1406).
</item 6>

<item 7>
A Hidden Treasure
Hidden Treasure (Arabic: کنزاً مخفیاً‎ ‎ ) is a Hadith Qudsi that has a very prominent role in Islamic mysticism and Islamic philosophy.
</item 7>

<item 8>
Reynold A. Nicholson
Reynold Alleyne Nicholson, or R. A. Nicholson (18 August 1868 – 27 August 1945), was an eminent English orientalist, scholar of both Islamic literature and
Islamic mysticism and widely regarded as one of the greatest Rumi (Mevlana or Mawlana) scholars and translators in the English language.
</item 8>

<item 9>
Ovamir Anjum
Ovamir Anjum is the Imam Khattab Chair of Islamic Studies at the Department of Philosophy, University of Toledo.

 He studies the connections between theology, ethics, politics, and law in classical and medieval Islam, with a subfocus on its comparisons with western
thought.

 Related fields of study include Islamic philosophy and Sufism.
</item 9>

<item 10>
Lawrencepur, Punjab, Pakistan
Hazrat Baba Jee Sahib is Qutb, Qutb of this time, Sufi Mystic ( Islamic mysticism ), Wali of Allah , acknowledged by various Islamic Scholars, Sufi Masters ,
Sufi Saints from all over the world.
</item 10>
=== END ITEM id=fd91c285575c6ef2 ===

=== ITEM id=e8393409a60f3b59 ===
Gold sentences and facts:
Sentence 0: Ionolyce is a small genus of butterflies in the family Lycaenidae.
  Fact 0.0: Ionolyce is a small genus of butterflies.
  Fact 0.1: Ionolyce is in the family Lycaenidae.
Sentence 1:  "I. helicon" is widespread in the Indoaustralasian region (India to the Malay Archipelago.
  Fact 1.0: I. helicon is widespread in the Indoaustralasian region.
  Fact 1.1: I. helicon is widespread from India to the Malay Archipelago.
Sentence 2: India lies on the Indian Plate, the northern portion of the Indo-Australian Plate, whose continental crust forms the Indian subcontinent.
  Fact 2.0: India lies on the Indian Plate.
  Fact 2.1: The Indian Plate is the northern portion of the Indo-Australian Plate.
  Fact 2.2: The continental crust of the Indian Plate forms the Indian subcontinent.
Sentence 3:  It is the seventh-largest country in the world, with a total area of 3287263 km2 .
  Fact 3.0: India is the seventh-largest country in the world.
  Fact 3.1: India has a total area of 3287263 km2.

Retrieved items:
<item 1>
Aulacodes peribocalis
Aulacodes peribocalis is a moth in the Crambidae family.

 It was described by Walker in 1859.

 It is found in Yemen and India.
</item 1>

<item 2>
Genoplesium
Genoplesium commonly known as midge orchids, is a genus of about 50 species of flowering plants in the orchid family, Orchidaceae and is found in Australia, New
Zealand and New Caledonia.

 Midge orchids are terrestrial herbs with a single leaf at the base of the plant.

 They are similar to orchids in the genus "Prasophyllum" in that plants without flowers have a hollow, onion-like leaf.

 The flowers are small but often scented and attractive to their insect pollinators.

 There is disagreement about which species belong to this genus and some taxonomists suggest that most belong in the genus "Corunastylis".
</item 2>

<item 3>
Leycesteria formosa
Leycesteria formosa (Himalayan honeysuckle, flowering nutmeg, Himalaya nutmeg or pheasant berry) is a deciduous shrub in the family Caprifoliaceae, native to
the Himalaya and southwestern China.

 It is considered a noxious invasive species in Australia, New Zealand, the neighboring islands of Micronesia, and some other places.
</item 3>

<item 4>
Consul fabius
Consul fabius, the tiger leafwing, is the most common and well known species of the genus "Consul" of subfamily Charaxinae in the brush-footed butterfly family
(Nymphalidae).

 It is found all over the Neotropics.
</item 4>

<item 5>
Kallima paralekta
Kallima paralekta, the Indian leafwing or Malayan leafwing, is a species of brush-footed butterfly of the genus "Kallima".

 Despite its common names, it is not found in India or Malaysia, but is endemic to Java and Sumatra of Indonesia.

 Like other members of its genus, it is remarkable for its strong resemblance to a dead leaf when its wings are folded.

 It was one of the species encountered by the British naturalist Alfred Russel Wallace in his travels in maritime Southeast Asia.

 It is mentioned in his famous 19th-century work "The Malay Archipelago" as one of the best examples of protective camouflage achieved through natural
selection.
</item 5>

<item 6>
White-bellied musk deer
The White-bellied musk deer or Himalayan musk deer ("Moschus leucogaster") is a musk deer species occurring in the Himalayas of Nepal, Bhutan, India, Pakistan
and China.

 It is listed as endangered by the IUCN because of overexploitation resulting in a probable serious population decline.
</item 6>

<item 7>
Psychopsis
Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.

 It was formerly included in the massively paraphyletic "wastebin genus" "Oncidium".

 The genus as a whole is commonly called butterfly orchids, but some species of other orchid genera are also called thus.
</item 7>

<item 8>
Aulacodes traversalis
Aulacodes traversalis is a moth in the Crambidae family.

 It was described by Harrison Gray Dyar Jr. in 1914.

 It is found in Panama.
</item 8>

<item 9>
Ionolyce
Ionolyce is a small genus of butterflies in the family Lycaenidae.

 "I. helicon" is widespread in the Indoaustralasian region (India to the Malay Archipelago.

 "I selkon"and "I.brunnescens" are both endemic to the Solomon Islands.
</item 9>

<item 10>
Ornithodoros savignyi
Ornithodoros savignyi aka Sand tampan or African eyed tampan or Kalahari sand tampan, is one of some 37 species in the genus "Ornithodoros" and is a soft tick
with a leathery, mammillated integument, causing paralysis and tampan toxicosis, two unrelated conditions.

 The sand tampan is an ectoparasite on humans, their livestock and wild animals, including birds and bats.

 Occurring in semi-desert areas of Africa, Saudi Arabia and other parts of the Persian Gulf, India, Sri Lanka and into Asia, it is able to survive for lengthy
periods without feeding, spending most of its life burrowed under sand or loose soil, often in wait for animals that rest or sleep under trees or in the lee of
rocks, but also in places where people or their animals congregate such as marketplaces, places of worship, cattle kraals and village squares.

 The timing of its activity is geared to coincide with that of potential hosts, but hot sunny conditions are usually avoided.
</item 10>
=== END ITEM id=e8393409a60f3b59 ===

=== ITEM id=9ec65aa64f2fe2c1 ===
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
Upper Saxon German
Upper Saxon (German: "Obersächsisch" ) is an East Central German dialect spoken in much of the modern German State of Saxony and in the adjacent parts of
Saxony-Anhalt and Thuringia.

 Though colloquially called "Saxon" ("Sächsisch" ), it is not to be confused with the Low Saxon dialect group in Northern Germany.

 Upper Saxon is closely linked to the Thuringian dialect spoken in the adjacent areas to the west.
</item 1>

<item 2>
Hundsheimer Berg
Hundsheimer Berg is a hill located in the Hundsheimer Berge hill range in Lower Austria, Austria close to the border with Slovakia.

 Its peak is 481 m above sea level which makes it the highest hill in the Hundsheimer Berge.

 The north slopes of the Hundsheimer Berg are covered with forests while the southern slopes have a steppe vegetation.

 Since 1965, a nature reserve with the same name, "Hundsheimer Berg", is located on these southern slopes.
</item 2>

<item 3>
Appenzeller cheese
Appenzeller cheese is a hard cow's-milk cheese produced in the Appenzell region of northeast Switzerland.

 A herbal brine, sometimes incorporating wine or cider, is applied to the wheels of cheese while they cure, which flavors and preserves the cheese while
promoting the formation of a rind.
</item 3>

<item 4>
List of Austrian mountain climbers
This is a list of Austrian mountaineers.

 On 5 of the 14 Eight-thousanders Austrians have made the first ascent, more than any other nation can claim.

 Out of the total of 9 Austrian mountaineers who made first ascents of Eight-thousanders, 3 have been members of the Edelweiss Club Salzburg, an association of
mountaineers founded in Salzburg in 1881
</item 4>

<item 5>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
</item 5>

<item 6>
Aspen/Snowmass
Aspen Snowmass is a winter resort complex located in Pitkin County in western Colorado in the United States.

 Owned and operated by the Aspen Skiing Company it comprises four skiing/snowboarding areas on four adjacent mountains in the vicinity of the towns of Aspen and
Snowmass Village.

 The four areas collectively form one of the most famous winter resorts in the world and are annually the destination for visitors from all over the world.
</item 6>

<item 7>
Appenzell Alps
The Appenzell Alps (German: "Appenzeller Alpen" ) are a mountain range in Switzerland on the northern edge of the Alps.

 They extend into the cantons of Appenzell Ausserrhoden, Appenzell Innerrhoden and St. Gallen and are bordered by the Glarus Alps to the west and the Rätikon to
the south-east.
</item 7>

<item 8>
South Bergish
South Bergish (German: "Südbergische Dialekte ") or Upper Bergish (German: "Oberbergische Dialekte ") is a group of German dialects of the Bergisches Land
Region East of the Rhine and approximately south of the Wupper and north of the Sieg.

 These dialects are part of the Ripuarian group and thus are also called East Ripuarian.

 Ripuarian dialects are also spoken west of the Rhine up to the German border, and in some small areas next to the respective borders in Belgium and in the
Netherlands.

 Ripuarian Bergish dialects belong to the Middle German group, and thus are varieties of High German, where they belong to the northmost ones.

 In the North, they border to the East Bergish and the West Bergish aka Low Bergish language groups, which are part of the Low Franconian group like Dutch.
</item 8>

<item 9>
Salzkammergut
The Salzkammergut is a resort area located in Austria.

 It stretches from the City of Salzburg eastwards along the Austrian Alpine Foreland and the Northern Limestone Alps to the peaks of the Dachstein Mountains,
spanning the federal states of Upper Austria, Salzburg, and Styria.

 The main river of the region is the Traun, a right tributary of the Danube.

 The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the
precious salt mines of the Habsburg Monarchy.

 It is a UNESCO World Heritage Site.
</item 9>

<item 10>
Obertraun
Obertraun, Upper Austria is a village in the Salzkammergut, a region in Austria.

 It is located near the Hallstätter See (Hallstatt Lake) and Hoher Dachstein.

 Obertraun is a popular holiday destination offering activities such as skiing, snowboarding in the winter and mountain biking, swimming and kayaking in the
summer.
</item 10>
=== END ITEM id=9ec65aa64f2fe2c1 ===
