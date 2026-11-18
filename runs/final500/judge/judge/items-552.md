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

=== ITEM id=e8c8b62a4c784121 ===
Gold sentences and facts:
Sentence 0: Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant", his ten-novel fantasy series.
  Fact 0.0: Stephen Reeder Donaldson was born on May 13, 1947.
  Fact 0.1: Stephen Reeder Donaldson is an American novelist.
  Fact 0.2: Stephen Reeder Donaldson writes fantasy novels.
  Fact 0.3: Stephen Reeder Donaldson writes science fiction novels.
  Fact 0.4: Stephen Reeder Donaldson writes mystery novels.
  Fact 0.5: Stephen Reeder Donaldson is most famous for "The Chronicles of Thomas Covenant".
  Fact 0.6: "The Chronicles of Thomas Covenant" is a ten-novel fantasy series.
  Fact 0.7: "The Chronicles of Thomas Covenant" is a series by Stephen Reeder Donaldson.
Sentence 1: Michael John Moorcock (born 18 December 1939) is an English writer, primarily of science fiction and fantasy, who has also published literary novels.
  Fact 1.0: Michael John Moorcock was born on 18 December 1939.
  Fact 1.1: Michael John Moorcock is an English writer.
  Fact 1.2: Michael John Moorcock is primarily a writer of science fiction.
  Fact 1.3: Michael John Moorcock is primarily a writer of fantasy.
  Fact 1.4: Michael John Moorcock has also published literary novels.

Retrieved items:
<item 1>
Epic Pooh
"Epic Pooh" is a 1978 article by the British science fiction writer Michael Moorcock, which reviews the field of epic fantasy, with a particular focus on epic
fantasy written for children.

 In it Moorcock critiques J. R. R. Tolkien's "The Lord of the Rings" for its politically conservative assumptions and for being escapist literature.
</item 1>

<item 2>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 2>

<item 3>
A Cure for Cancer
A Cure for Cancer is a novel by British fantasy and science fiction writer Michael Moorcock, first published in London 1971 by Allison and Busby.

 The book is part of Moorcock's long-running Jerry Cornelius series.
</item 3>

<item 4>
Michael Moorcock
Michael John Moorcock (born 18 December 1939) is an English writer, primarily of science fiction and fantasy, who has also published literary novels.

 He is best known for his novels about the character Elric of Melniboné, a seminal influence on the field of fantasy in the 1960s and 1970s.
</item 4>

<item 5>
Douglas Hill
Douglas Arthur Hill (6 April 1935 – 21 June 2007) was a Canadian science fiction author, editor and reviewer.

 He was born in Brandon, Manitoba, the son of a railroad engineer, and was raised in Prince Albert, Saskatchewan.

 An avid science fiction reader from an early age, he studied English at the University of Saskatchewan (where he earned an Honours B.A. in 1957) and at the
University of Toronto.

 He married fellow writer and U. of S. alumna Gail Robinson (writer) in 1958; they moved to Britain in 1959, where he worked as a freelance writer and editor
for Aldus Books.

 In 1967–1968 he served as Assistant Editor of the controversial "New Worlds" science fiction magazine under Michael Moorcock.
</item 5>

<item 6>
Robert Holdstock
Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish
fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
</item 6>

<item 7>
Behold the Man (novel)
Behold the Man (1969) is a science fiction novel by British writer Michael Moorcock.

 It originally appeared as a novella in a 1966 issue of "New Worlds"; later, Moorcock produced an expanded version which was first published in 1969 by Allison
& Busby.

 The title derives from the Gospel of John, Chapter 19, Verse 5: "Then Jesus came out, wearing the crown of thorns and the purple robe.

 And Pilate said to them Behold the Man."
</item 7>

<item 8>
Warlord of the Air
The Warlord of the Air is a 1971 British alternate history science fiction novel written by Michael Moorcock.

 It concerns the adventures of Oswald Bastable, an Edwardian era soldier stationed in India, and his adventures in an alternate universe, in his own future,
wherein the First World War never happened.

 It is the first part of Moorcock's 'A Nomad of the Time Streams' trilogy and, in its use of speculative technology (such as airships) juxtaposed against an
Edwardian setting, it is widely considered to be one of the first steampunk novels.

 The novel was first published by Ace Books as part of their Ace Science Fiction Specials series.
</item 8>

<item 9>
Edgar Allan Poe’s humorous “A Predicament,” originally “The Scythe of Time,” parodies Gothic sensation tales and is often paired with “How to Write a Blackwood
Article.” British writer Michael Moorcock (born 1939), famed for Elric, wrote the science-fiction novels *Behold the Man* (expanded from a 1966 novella; title
from John’s Gospel), *The Final Programme*, *The Warlord of the Air* (1971; an
</item 9>

<item 10>
The Final Programme
The Final Programme is a novel by British science fiction and fantasy writer Michael Moorcock.

 Written in 1965 as the underground culture was beginning to emerge, it was not published for several years.

 Moorcock has stated that publishers at the time considered it was "too freaky".
</item 10>
=== END ITEM id=e8c8b62a4c784121 ===

=== ITEM id=dcd4c5ff265e6d12 ===
Gold sentences and facts:
Sentence 0: The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.
  Fact 0.0: The Dundee Canal was an industrial canal.
  Fact 0.1: The Dundee Canal was in Clifton and Passaic.
  Fact 0.2: Clifton and Passaic are in Passaic County, New Jersey.
Sentence 1:  It was built between 1858 and 1861 and ran parallel to the Passaic River.
  Fact 1.0: The Dundee Canal was built between 1858 and 1861.
  Fact 1.1: The Dundee Canal ran parallel to the Passaic River.
Sentence 2:  It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.
  Fact 2.0: It runs parallel to the coast.
  Fact 2.1: It is separated from the Atlantic Ocean by barrier beaches.
  Fact 2.2: The barrier beaches include Palm Beach Island.
Sentence 3:  The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
  Fact 3.0: The lagoon is connected to the Atlantic Ocean.
  Fact 3.1: The lagoon is connected to the Atlantic Ocean by two inlets.
  Fact 3.2: The two inlets are permanent.
  Fact 3.3: The two inlets are man-made.

Retrieved items:
<item 1>
Lake Worth, Florida
Lake Worth is a city in Palm Beach County, Florida, United States, which takes its name from the body of water along its eastern border, originally called "Lake
Worth", and now generally known as the Lake Worth Lagoon.

 The lake itself was named for General William J. Worth, who led U.S. forces during the last part of the Second Seminole War.

 s of 2010 , the population estimated by the U.S. Census Bureau was 34,910.

 The city is included in the Miami-Fort Lauderdale-Pompano Beach Metropolitan Area, which is home to approximately 5,563,857 people.
</item 1>

<item 2>
Lake Worth Lagoon
The Lake Worth Lagoon is a lagoon located in Palm Beach County, Florida.

 It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.

 The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
</item 2>

<item 3>
Lake Worth station
Lake Worth is a Tri-Rail commuter rail station in Lake Worth, Florida, at the confluence of Lake Worth Road (SR 802) and Interstate 95.

 Opening to service January 9, 1989, parking is available at this station, all of which is beneath I-95 on the south side of Lake Worth Road.
</item 3>

<item 4>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 4>

<item 5>
South Lake Worth Inlet
The South Lake Worth Inlet, also known as the Boynton Inlet, is an artificial cut through a barrier beach connecting the south end of the Lake Worth Lagoon in
Palm Beach County, Florida with the Atlantic Ocean.

 The inlet is 130 ft wide and 6 ft and 12 ft deep.
</item 5>

<item 6>
Lake Worth Community High School
Lake Worth Community High School is a B State of Florida rated public high school located in Lake Worth, Florida.

 Established in 1922 as Lake Worth High School, it is currently one of Palm Beach County's largest schools.

 It has six magnet programs: the award-winning Air Force JROTC led by former Multi-National Award-winning Drill Teams, Tyranny and Fidelis; Medicine and Allied
Health; Child Care and Development; Criminal Justice; Culinary and Drafting and Design.

 The Palm Beach County School Board added the word Community to the names of all public high schools, including Lake Worth High School, in the 1980s.
</item 6>

<item 7>
Lake Worth Inlet
The Palm Beach Inlet, also known as the Lake Worth Inlet is an artificial cut through a barrier island connecting the northern part of the Lake Worth Lagoon in
Palm Beach County, Florida with the Atlantic Ocean.

 It is bordered by the town of Palm Beach on the south, and by the town of Palm Beach Shores to the north.

 The inlet is also the entrance channel for the Port of Palm Beach.

 Its coordinates are .
</item 7>

<item 8>
Raritan Water Power Canal
The Raritan Water Power Canal in Raritan, New Jersey was a three-mile long water power canal built during the early 1840s to power industries.

 The canal has been determined to be eligible for the National Register.

 The canal is included in Duke Island Park, as part of the Somerset County Park System.
</item 8>

<item 9>
Lake Osborne
Lake Osborne, Florida, USA is a 378-acre (152.9 hectares) lake that is part of a system of once natural freshwater lakes lying along the western slope of the
coastal ridge in Palm Beach County just west of the Florida Intracoastal Waterway and Atlantic Ocean.

 It is located within the C-16 drainage basin which occupies approximately 40,031 acres of land (16,200 hectares).

 Five drainage canals discharge directly to Lake Osborne, and Lake Osborne discharges via the E-4 canal to the C16 and C51 canals to the Lake Worth Lagoon.

 The lake is bordered on the west by John Prince Memorial Park, and on the east be the City of Lake Worth.
</item 9>

<item 10>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 10>
=== END ITEM id=dcd4c5ff265e6d12 ===

=== ITEM id=3c3d1d8f6694708a ===
Gold sentences and facts:
Sentence 0: John André (2 May 1750 – 2 October 1780) was a British Army officer hanged as a spy by the Continental Army during the American Revolutionary War for assisting Benedict Arnold's attempted surrender of the fort at West Point, New York to the British.
  Fact 0.0: John André lived from 2 May 1750 to 2 October 1780.
  Fact 0.1: John André was a British Army officer.
  Fact 0.2: John André was hanged as a spy by the Continental Army during the American Revolutionary War.
  Fact 0.3: John André was hanged for assisting Benedict Arnold's attempted surrender of the fort at West Point, New York, to the British.
Sentence 1: Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American Continental Army, and later defected to the British Army.
  Fact 1.0: Benedict Arnold lived from January 14, 1741 (Old Style: January 3, 1740) to June 14, 1801.
  Fact 1.1: Benedict Arnold was a general during the American Revolutionary War.
  Fact 1.2: Benedict Arnold fought for the American Continental Army.
  Fact 1.3: Benedict Arnold later defected to the British Army.

Retrieved items:
<item 1>
John André
John André (2 May 1750 – 2 October 1780) was a British Army officer hanged as a spy by the Continental Army during the American Revolutionary War for assisting
Benedict Arnold's attempted surrender of the fort at West Point, New York to the British.
</item 1>

<item 2>
List of Continental Army units
The Continental Army was the national army of first the Thirteen Colonies, and then the independent United States, during the American Revolutionary War,
established by a resolution of the Congress on June 14, 1775, three days before the Battle of Bunker Hill, where it saw its first action under that title.

.

 The Continental Congress took a number of steps in the spring of 1775 to create the army in response to the Battles of Lexington and Concord in April and the
seizure of Fort Ticonderoga in May.

 The units composing the Continental Army changed frequently, especially in the first two years of the war.

 From 1777 to the close of the war, the organization of the Continental Army became progressively more systematic and sophisticated.

 The Continental Army that served at Yorktown in 1781 bore very little resemblance to the Continental Army that blockaded Boston in 1775.
</item 2>

<item 3>
William Farrar Smith
William Farrar Smith (February 17, 1824 – February 28, 1903), known as ‘Baldy’ Smith, was a Union general in the American Civil War, notable for attracting the
extremes of glory and blame.

 He was praised for his gallantry in the Seven Days Battles and the Battle of Antietam, but was demoted for insubordination after the disastrous defeat at the
Battle of Fredericksburg.

 As chief engineer of the Army of the Cumberland, he achieved recognition by restoring a supply line that saved that army from starvation and surrender, known
the “Cracker Line”, that helped Union troops to success in the Chattanooga Campaign in the autumn of 1863.

 Leading the first operation against Petersburg, Smith’s hesitation, possibly illness-related, cost the Union a prime opportunity for a quick end to the war,
and he was relieved of command.
</item 3>

<item 4>
Battle of Germantown
The Battle of Germantown was a major engagement in the Philadelphia campaign of the American Revolutionary War.

 It was fought on October 4, 1777, at Germantown, Pennsylvania, between the British Army led by Sir William Howe, and the American Continental Army, with the
2nd Canadian Regiment, under George Washington.
</item 4>

<item 5>
Lewis Nicola
Lewis Nicola (1717 – August 9, 1807) was an Irish-born American military officer, merchant, and writer who held various military and civilian positions
throughout his career.

 Nicola is most notable for authoring the Newburgh letter, which urged George Washington to assume a royal title.

 Born in Dublin, Ireland, Nicola had been an officer in the British Army, serving in Europe before immigrating to the Thirteen Colonies.

 Establishing a residence in Philadelphia with his family, Nicola opened a library in 1767 and was active in colonial philosophical organizations.

 As a result of his work to establish the American Philosophical Society, he was elected as one of its curators.

 When the American Revolution broke out, Nicola offered his services to the colonial government, which eventually appointed him to various positions with local
forces.
</item 5>

<item 6>
John F. Anderson
John Francis Anderson was an officer in the Union Army during the American Civil War.

 When the war broke out, Anderson who was the son of Maine's former Governor Hugh J. Anderson was initially commissioned First Lieutenant in the militia; and on
September 2, 1861, Anderson became the adjutant of the 24th Regiment Massachusetts Volunteer Infantry.

 He was appointed to Maj. Gen. John G. Foster's staff as a major and an aide-de-camp on June 9, 1863 and served with that officer for most of the remainder of
the war, resigning on March 27, 1865.

 On December 11, 1866, President Andrew Johnson nominated Major Anderson to the honorary grade of brevet brigadier general, to rank from March 13, 1865 and the
United States Senate confirmed the award on February 6, 1867.
</item 6>

<item 7>
Benedict Arnold
Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American
Continental Army, and later defected to the British Army.

 While a general on the American side, he obtained command of the fortifications at West Point, New York (which after 1802 would become the site of the U.S.
Military Academy), overlooking the cliffs at the Hudson River (upriver from British-occupied New York City), and planned to surrender them to British forces.

 This plan was exposed in September 1780.

 He was commissioned into the British Army as a brigadier general.
</item 7>

<item 8>
Benjamin Logan
Benjamin Logan (c.1742 – December 11, 1802) was an American pioneer, soldier, and politician from Shelby County, Kentucky.

 As colonel of the Kentucky County, Virginia militia during the American Revolutionary War, he was second-in-command of all the militia in Kentucky.

 He was also a leader in Kentucky's efforts to become a state.

 His brother, John Logan, was the first state treasurer of Kentucky.
</item 8>

<item 9>
Siege of Yorktown
The Siege of Yorktown, also known as the Battle of Yorktown, the Surrender at Yorktown, German Battle or the Siege of Little York, ending on October 19, 1781,
at Yorktown, Virginia, was a decisive victory by a combined force of American Continental Army troops led by General George Washington and French Army troops
led by the Comte de Rochambeau over a British Army commanded by British peer and Lieutenant General Charles Cornwallis.

 The culmination of the Yorktown campaign, the siege proved to be the last major land battle of the American Revolutionary War in the North American theater, as
the surrender by Cornwallis, and the capture of both him and his army, prompted the British government to negotiate an end to the conflict.

 The battle boosted faltering American morale and revived French enthusiasm for the war, as well as undermining popular support for the conflict in Great
Britain.
</item 9>

<item 10>
Benjamin Lincoln
Benjamin Lincoln (January 24, 1733 (O.S. January 13, 1732) – May 9, 1810) was an American army officer.

 He served as a major general in the Continental Army during the American Revolutionary War.

 Lincoln is notable for being involved in three major surrenders during the war: his participation in the Battles of Saratoga (sustaining a wound shortly
afterward) contributed to John Burgoyne's surrender of a British army, he oversaw the largest American surrender of the war at the 1780 Siege of Charleston,
and, as George Washington's second in command, he formally accepted the British surrender at Yorktown.
</item 10>
=== END ITEM id=3c3d1d8f6694708a ===

=== ITEM id=33ed88af214d31f4 ===
Gold sentences and facts:
Sentence 0:  It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke Bader and Linda Riley.
  Fact 0.0: The publication was launched in March 1994.
  Fact 0.1: The publication was launched by Millivres Ltd.
  Fact 0.2: The publication was launched under the editorship of Frances Williams.
  Fact 0.3: The publication is now published by Twin Media Group.
  Fact 0.4: Twin Media Group is owned by Silke Bader.
  Fact 0.5: Twin Media Group is owned by Linda Riley.
Sentence 1: Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.
  Fact 1.0: Elle is a worldwide lifestyle magazine.
  Fact 1.1: Elle is of French origin.
  Fact 1.2: Elle focuses on fashion.
  Fact 1.3: Elle focuses on beauty.
  Fact 1.4: Elle focuses on health.
  Fact 1.5: Elle focuses on entertainment.
Sentence 2:  "Elle" is also the world's best-selling fashion magazine.
  Fact 2.0: Elle is also the world's best-selling fashion magazine.
Sentence 3:  It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.
  Fact 3.0: Elle was founded by Pierre Lazareff.
  Fact 3.1: Elle was founded by Hélène Gordon, the wife of Pierre Lazareff.
  Fact 3.2: Elle was founded in 1945.

Retrieved items:
<item 1>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 1>

<item 2>
Rester vrai
Rester vrai is a 1994 album recorded by French singer Florent Pagny.

 It was his third studio album and was released in March 1994.

 It achieved moderate success in France where it remained charted for 39 weeks in the top 50, including a peak at #19 for two weeks.

 For the first time in Pagny's career, lyrics were written by other artists such as Jean-Jacques Goldman who participated under the pseudonym of Sam Brewski.

 This album provided two singles : the unsuccessful "Est-ce que tu me suis ?"

 (#45 in France) and the hit single "Si tu veux m'essayer" (#7 in France), which was recorded as "If You Want to Know Me" in 2009 by the Belgian singer Dana
Winner for the album "Between Now and Tomorrow" with English lyrics by Michael Leahy.

 The music video for the track "Rester vrai" was directed by Dani Jacobs and shot in London and features Pagny trapped in a glass box in a large empty space as
if he is a museum exhibit or part of an experiment.
</item 2>

<item 3>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 3>

<item 4>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 4>

<item 5>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 5>

<item 6>
Katarina Ivanovska
Katarina Ivanovska (Macedonian: Катарина Ивановска ; born 18 August 1988) is a Macedonian model and actress.

 She began her modeling career in 2004, appearing at Milan Fashion Week after winning the Look Models International model search in Macedonia.

 In December, 2004, she appeared in a pictorial for "Elle" magazine and has also appeared in "Citizen K", "Stiletto" and the Italian and Russian "Vogue".

 She has been featured on the covers of "Diva" and "Maxima" magazines and in advertisements for D&G in 2006.

 She is considered the most successful Macedonian model.

 In 2010, Ivanovska appeared in Serbian "Elle" magazine.

 In 2011 she signed a contract for advertising Victoria's Secret products.

 In 2011 she got her first acting job in the Macedonian World War II film, "The Third Half", landing the lead role of a young Jewish girl named Rebecca.
</item 6>

<item 7>
The Fader
The Fader (stylized as The FADER) is a New York City-based music magazine launched in 1999 by Rob Stone and Jon Cohen, covering music, style and culture.

 It was the first print publication to be released on iTunes.
</item 7>

<item 8>
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
</item 8>

<item 9>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 9>

<item 10>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 10>
=== END ITEM id=33ed88af214d31f4 ===

=== ITEM id=7d338361fe061ffc ===
Gold sentences and facts:
Sentence 0:  It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black matipo", "Pittosporum tenuifolium", is not related to "Myrsine").
  Fact 0.0: Myrsine contains about 200 species.
  Fact 0.1: Myrsine includes several notable radiations.
  Fact 0.2: The matipo of New Zealand is one of the notable radiations within Myrsine.
  Fact 0.3: The kōlea of Hawaiʻi is one of the notable radiations within Myrsine.
  Fact 0.4: The New Zealand 'black matipo', Pittosporum tenuifolium, is not related to Myrsine.
Sentence 1: Storax or snowbell is the common names of Styrax, a genus of about 130 species of large shrubs or small trees in the family Styracaceae, mostly native to warm temperate to tropical regions of the Northern Hemisphere, with the majority in eastern and southeastern Asia, but also crossing the equator in South America.
  Fact 1.0: Storax or snowbell are the common names of Styrax.
  Fact 1.1: Styrax is a genus of about 130 species.
  Fact 1.2: Styrax species are large shrubs or small trees.
  Fact 1.3: Styrax belongs to the family Styracaceae.
  Fact 1.4: Styrax is mostly native to warm temperate to tropical regions of the Northern Hemisphere.
  Fact 1.5: The majority of Styrax species are in eastern and southeastern Asia.
  Fact 1.6: Styrax also crosses the equator in South America.

Retrieved items:
<item 1>
Styrax litseoides
Styrax litseoides is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is endemic to Vietnam.
</item 1>

<item 2>
Styrax crotonoides
Styrax crotonoides is a species of plant in the genus "Styrax" and family Styracaceae.

 It is found in Peninsular Malaysia and Singapore.
</item 2>

<item 3>
Rapanea
Rapanea is a genus of plant in family Primulaceae.

 It has often been placed in synonymy with "Myrsine", and many species have been moved to "Myrsine".
</item 3>

<item 4>
Styrax fraserensis
Styrax fraserensis is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is a tree endemic to Peninsular Malaysia.

 It is threatened by habitat loss.
</item 4>

<item 5>
Styrax vilcabambae
Styrax vilcabambae (syn.

 "Pamphilia vilcabambae" D.R.Simpson) is a species of flowering plant in the genus "Styrax" and the family Styracaceae.

 It is endemic to Peru.
</item 5>

<item 6>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 6>

<item 7>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 7>

<item 8>
Ardisia
Ardisia (coralberry or marlberry) is a genus of flowering plants in the family Primulaceae.

 It was in the former Myrsinaceae family now recognised as the myrsine sub-family Myrsinoideae.

 They are distributed in the Americas, Asia, Australia, and the Pacific Islands, mainly in the tropics.

 There are 400 to 500 species.
</item 8>

<item 9>
Styrax
Storax or snowbell is the common names of Styrax, a genus of about 130 species of large shrubs or small trees in the family Styracaceae, mostly native to warm
temperate to tropical regions of the Northern Hemisphere, with the majority in eastern and southeastern Asia, but also crossing the equator in South America.

 The resin obtained from the tree is called storax or benzoin (not to be confused with the "Liquidambar" storax balsam).
</item 9>

<item 10>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 10>
=== END ITEM id=7d338361fe061ffc ===

=== ITEM id=5796eeaee1d9f8ad ===
Gold sentences and facts:
Sentence 0: Mosè in Egitto ("Moses in Egypt)" (] ) is a three-act opera written by Gioachino Rossini to an Italian libretto by Andrea Leone Tottola, which was based on a 1760 play by Francesco Ringhieri, "L'Osiride".
  Fact 0.0: Mosè in Egitto (Moses in Egypt) is a three-act opera.
  Fact 0.1: Mosè in Egitto was written by Gioachino Rossini.
  Fact 0.2: Mosè in Egitto has an Italian libretto by Andrea Leone Tottola.
  Fact 0.3: The libretto of Mosè in Egitto was based on a 1760 play by Francesco Ringhieri.
  Fact 0.4: The 1760 play by Francesco Ringhieri is titled L'Osiride.
Sentence 1: Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.
  Fact 1.0: Lucrezia Borgia is a melodramatic opera.
  Fact 1.1: Lucrezia Borgia is in a prologue and two acts.
  Fact 1.2: Lucrezia Borgia is by Gaetano Donizetti.

Retrieved items:
<item 1>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 1>

<item 2>
Mosè in Egitto
Mosè in Egitto ("Moses in Egypt)" (] ) is a three-act opera written by Gioachino Rossini to an Italian libretto by Andrea Leone Tottola, which was based on a
1760 play by Francesco Ringhieri, "L'Osiride".

 It premièred on 5 March 1818 at the recently reconstructed Teatro San Carlo in Naples, Italy.
</item 2>

<item 3>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 3>

<item 4>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 4>

<item 5>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 5>

<item 6>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 6>

<item 7>
L'Orfeo discography
These lists show the audio and visual recordings of the opera "L'Orfeo" by Claudio Monteverdi.

 The opera was first performed in Mantua in 1607, at the court of Duke Vincenzo Gonzaga, and is one of the earliest of all operas.

 The first recording of "L'Orfeo" was issued in 1939, a freely adapted version of Monteverdi's music edited by Giacomo Benvenuti, given by the orchestra of La
Scala Milan conducted by Ferrucio Calusio.

 In 1949 the Berlin Radio Orchestra under Helmut Koch recorded the complete opera, on long-playing records (LPs).

 The advent of LP recordings was, as Harold Schonberg later wrote, an important factor in the postwar revival of interest in Renaissance and Baroque music, and
from the mid-1950s recordings of "L'Orfeo" have been issued on many labels.

 Koch's landmark version was reissued in 1962, when it was compared unfavourably with others that had by then been issued.
</item 7>

<item 8>
Lucrezia Borgia (1940 film)
Lucrezia Borgia is a 1940 Italian historical film directed by Hans Hinrich and starring Isa Pola, Friedrich Benfer and Carlo Ninchi.

 The film portrays the life of Lucrezia Borgia (1480-1519), one of a number of Italian films of the era set during the Renaissance.

 It was made at the Scalera Studios in Rome.
</item 8>

<item 9>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 9>

<item 10>
Lucrezia Borgia (play)
Lucrezia Borgia (French: "Lucrèce Borgia" ) is an 1833 play by the French writer Victor Hugo.

 It is a historical work portraying the Renaissance-era Italian aristocrat Lucrezia Borgia.

 The play (along with "Angelo, Tyrant of Padua") is believed to have been a major influence on Oscar Wilde's "The Duchess of Padua" (1891).
</item 10>
=== END ITEM id=5796eeaee1d9f8ad ===

=== ITEM id=daf0501b54e9235d ===
Gold sentences and facts:
Sentence 0: Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.
  Fact 0.0: Lowell J. Sherman was born on October 11, 1885.
  Fact 0.1: Lowell J. Sherman died on December 28, 1934.
  Fact 0.2: Lowell J. Sherman was an American actor.
  Fact 0.3: Lowell J. Sherman was an American film director.
Sentence 1: Jonathan Kaplan (born November 25, 1947) is an American film producer and director.
  Fact 1.0: Jonathan Kaplan was born on November 25, 1947.
  Fact 1.1: Jonathan Kaplan is an American film producer.
  Fact 1.2: Jonathan Kaplan is an American film director.

Retrieved items:
<item 1>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 1>

<item 2>
Lowell Sherman
Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.

 At a time when it was highly unusual, he was both the actor and director on several films in the early 1930s, before completely transitioning to the role of
director.

 At the height of his career, after scoring huge successes with his direction of the films "She Done Him Wrong" and "Morning Glory" (which starred Mae West and
won her first Academy Award for Katharine Hepburn respectively), he succumbed to pneumonia after a brief illness.
</item 2>

<item 3>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 3>

<item 4>
D. Ross Lederman
David Ross Lederman (December 12, 1894 – August 24, 1972) was an American film director noted for his Western/action/adventure films of the 1930s and 1940s.
</item 4>

<item 5>
Maurice Browne
Maurice Browne (12 February 1881 – 21 January 1955), born in Reading, England, was best known as a theater producer in the United States and the UK.

 The Cambridge-educated Browne was also a poet, actor, and theater director.

 He has been credited, along with his then-wife Ellen Van Volkenburg, with being the founder of the Little Theatre Movement in America through his work with the
Chicago Little Theatre.

 Browne and Van Volkenburg went on to found the department of drama at the Cornish School in Seattle in 1918, now Cornish College of the Arts.

 Browne's greatest triumph came in 1929 when he produced "Journey's End", by R. C. Sherriff in London.
</item 5>

<item 6>
Jonathan Kaplan
Jonathan Kaplan (born November 25, 1947) is an American film producer and director.

 His film "The Accused" (1988) earned actress Jodie Foster her first Oscar for Best Actress and was nominated for the Golden Bear at the 39th Berlin
International Film Festival.

 Kaplan received five Emmy nominations for his roles directing and producing "ER".
</item 6>

<item 7>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 7>

<item 8>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 8>

<item 9>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 9>

<item 10>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 10>
=== END ITEM id=daf0501b54e9235d ===

=== ITEM id=d101d90eb53606a9 ===
Gold sentences and facts:
Sentence 0: Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.
  Fact 0.0: Old Smokey is a euphemistic name for the state prison electric chair in New Jersey.
  Fact 0.1: The New Jersey state prison electric chair, called Old Smokey, is on display at the New Jersey State Police Museum.
Sentence 1:  The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
  Fact 1.0: The chair's most notorious victim was Richard Hauptmann.
  Fact 1.1: Richard Hauptmann was executed in the chair.
  Fact 1.2: Richard Hauptmann was found guilty of abducting and killing Charles Augustus Lindbergh, Jr.
  Fact 1.3: The abduction and killing of Charles Augustus Lindbergh, Jr. happened in 1932.
  Fact 1.4: Charles Augustus Lindbergh, Jr. was 20 months old.
  Fact 1.5: Charles Augustus Lindbergh, Jr. was the son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh.
  Fact 1.6: The case is referred to as the Lindbergh kidnapping (see Lindbergh kidnapping).
Sentence 2:  The Lindbergh kidnapping became known as "The Crime of the Century".
  Fact 2.0: The Lindbergh kidnapping became known as 'The Crime of the Century'.

Retrieved items:
<item 1>
Johnny Jack Nounes
Johnny Jack Nounes, also known as the "Beau Brummell of Galveston", was an organized crime boss in Galveston, Texas, United States, during the early 1900s.

 He, with one-armed George Musey, led the Downtown Gang, one of the two gangs which controlled most of the Galveston Crime Syndicate until the early 1930s.

 They fought for control of the island against the rival Beach Gang led by Ollie Quinn and Dutch Voight.

 As the prohibition era began, his gang came to be one of the dominant forces in the Galveston Crime Syndicate.

 Galveston became the main port of entry for liquor supply in Texas and many parts of the Midwest.
</item 1>

<item 2>
Smiley face murder theory
The Smiley face murder theory (variations include Smiley face murders, Smiley face killings, Smiley face gang, and others) is a theory advanced by two retired
New York City detectives, Kevin Gannon and Anthony Duarte, that a number of young men found dead in bodies of water across several Midwestern American states
over the last decade did not accidentally drown, as concluded by law enforcement agencies, but were victims of a serial killer or killers.

 The term "smiley face" became connected to the alleged murders when it was made public that the police had discovered graffiti depicting a smiley face near
locations where they think the killer dumped the bodies in at least a dozen of the cases.

 The response of law enforcement investigators and other experts to Gannon and Duarte's theory has been largely skeptical.
</item 2>

<item 3>
The Last Mile (play)
The Last Mile is a Broadway play by John Wexley that ran for 289 performances from February 13, 1930 to October 1930 at the Sam H. Harris Theatre.

 It was produced by Herman Shumlin and staged by Chester Erskine.

 It is set in the death row wing of a prison.

 The lead role of John "Killer" Mears was first played by Spencer Tracy, and it was the role that bought him to the attention of Hollywood.

 It was later played for a time by Clark Gable.

 The play was adapted into a 1932 film starring Preston Foster and into a 1959 film starring Mickey Rooney.
</item 3>

<item 4>
The Big House (1930 film)
The Big House is a 1930 American Pre-Code crime drama film directed by George W. Hill, released by Metro-Goldwyn-Mayer, and starring Chester Morris, Wallace
Beery, Lewis Stone and Robert Montgomery.

 The supporting cast features Leila Hyams, George F. Marion, J. C. Nugent, Karl Dane and Tom Kennedy.

 The story and dialogue were written by Frances Marion, with additional dialogue by Joe Farnham and Martin Flavin.

 The story was inspired by a spate of prison riots in 1929 and resulting federal investigation.

 In response, George Hill wrote a twenty-seven page story treatment called "The Reign of Terror: A Story of Crime and Punishment".

 Irving Thalberg gave the go ahead for the screenplay and assigned Frances Marion to work with George Hill.
</item 4>

<item 5>
Old Smokey
Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.

 The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus
Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
</item 5>

<item 6>
Sten Forshufvud
Sten Gabriel Bernhard Forshufvud (9 February 1903 – 25 June 1985) was a Swedish dentist and physician, and amateur toxicologist (expert on poisons) who
formulated and supported the controversial theory that Napoleon was assassinated by a member of his entourage while in exile.

 He wrote a book, in Swedish, about this in 1961, which was translated the following year as "Who Killed Napoleon?"

 He later published his ideas in English, in the 1983 book "Assassination At St. Helena: The Poisoning Of Napoleon Bonaparte," a book on whose authorship Ben
Weider, co-author (with David Hapgood) of the book "The Murder Of Napoleon," published the year earlier, which also advanced Forshufvud's ideas, collaborated.
</item 6>

<item 7>
Old Sparky
Old Sparky is the nickname of the electric chairs in Arkansas, Connecticut, Florida, Georgia, Illinois, Kentucky, Nebraska, New York, Ohio, Oklahoma, South
Carolina, Texas, Virginia, and West Virginia.

 Old Smokey was the nickname of the electric chairs used in New Jersey, Pennsylvania, and Tennessee.
</item 7>

<item 8>
Convicted (1950 film)
Convicted is a 1950 American crime film noir directed by Henry Levin starring Glenn Ford and Broderick Crawford.

 It was the third Columbia Pictures film adaptation of the 1929 stage play "The Criminal Code" by Martin Flavin, following Howard Hawk's "The Criminal Code"
(1931) and John Brahm's "Penitentiary" (1938).
</item 8>

<item 9>
The Journalist and the Murderer
The Journalist and the Murderer is a study by Janet Malcolm about the ethics of journalism, published by Alfred A. Knopf/Random House in 1990.

 It is an examination of the professional choices that shape a work of non-fiction, as well as a rumination on the morality that underpins the journalistic
enterprise.

 The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss'
1983 book "Fatal Vision".
</item 9>

<item 10>
Richard Hauptmann
Bruno Richard Hauptmann (November 26, 1899 – April 3, 1936) was a German-born carpenter who was convicted of the abduction and murder of the 20-month-old son of
aviator Charles Lindbergh and his wife Anne Morrow Lindbergh.

 The Lindbergh kidnapping became known as "The Crime of the Century".

 Hauptmann proclaimed his innocence to the end, but he was convicted of first degree murder and executed in 1936 in the electric chair at the New Jersey State
Prison.
</item 10>
=== END ITEM id=d101d90eb53606a9 ===

=== ITEM id=44fb8daeb48b37a0 ===
Gold sentences and facts:
Sentence 0: Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.
  Fact 0.0: Return to Never Land is also known as Peter Pan 2.
  Fact 0.1: Return to Never Land is also known as Peter Pan In: Return to Never Land.
  Fact 0.2: Return to Never Land is a 2002 American film.
  Fact 0.3: Return to Never Land is an animated musical fantasy-adventure film.
  Fact 0.4: Return to Never Land was produced by Walt Disney Television Animation.
  Fact 0.5: Return to Never Land was released by Walt Disney Pictures.
  Fact 0.6: Return to Never Land was distributed by Walt Disney Studios Motion Pictures.
Sentence 1:  The film was co-produced by Walt Disney Pictures and Mayhem Pictures.
  Fact 1.0: Return to Never Land was co-produced by Walt Disney Pictures.
  Fact 1.1: Return to Never Land was co-produced by Mayhem Pictures.

Retrieved items:
<item 1>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 1>

<item 2>
Josann McGibbon
Josann McGibbon is an American screenwriter working in partnership with Sara Parriott.

 The team's first major success as a screenwriter was the early Brad Pitt film, "The Favor".

 Their biggest hits since then include "Three Men and a Little Lady" and "Runaway Bride".

 In 2007, McGibbon and Parriott co-wrote and produced the hit Debra Messing miniseries, "The Starter Wife".

 "The Starter Wife" received 10 Emmy nominations in 2007, including for best screenwriting, and won one Emmy Award.

 It was also nominated for Golden Globe and Writers Guild awards, and was then produced as a series, also on USA Network.

 McGibbon and Parriott wrote and co-produced the Disney Channel movie, "Descendants" which was directed by Kenny Ortega and premiered in July, 2015.

 On February, 2013, it won the Writers Guild of America Award in television for Outstanding Children's Long Form.

 They then wrote and executive-produced "Descendants 2", also directed by Kenny Ortega, which was simulcast on the Disney Channel, ABC, and the other cable
channels owned by Disney-ABC in July, 2017.
</item 2>

<item 3>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 3>

<item 4>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 4>

<item 5>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 5>

<item 6>
Cash McCall
Cash McCall is a 1960 American romantic drama film in Technicolor from Warner Bros., produced by Henry Blanke, directed by Joseph Pevney, that stars James
Garner and Natalie Wood.

 It is based upon the novel of the same name by Cameron Hawley about a man who buys moribund businesses in order to refurbish them and then sell them on at
considerable profit.

 The film's screenplay is by Lenore J. Coffee and Marion Hargrove.
</item 6>

<item 7>
Peter Pan: Adventures in Never Land
Peter Pan: Adventures in Never Land (also known as Peter Pan in Disney's Return to Never Land) is a 2002 video game based on the Peter Pan Disney franchise.
</item 7>

<item 8>
Us (film)
Us is a 1991 television movie broadcast on CBS, produced, written and directed by Michael Landon.

 Landon also starred in the film, along with Barney Martin and Casey Peterson.

 It was a pilot for what would have been Landon's fourth consecutive television series; Landon's death that year precluded its going ahead.

 Landon played Jeff Hayes, a man just released from prison after serving many years due to being wrongfully convicted of killing a wealthy man's wife.
</item 8>

<item 9>
Return to Never Land
Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by
Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.

 The film is a sequel to Walt Disney Feature Animation 1953 film "Peter Pan", It is based on J. M. Barrie's novel "Peter and Wendy", and had a worldwide gross
of $109 million.
</item 9>

<item 10>
Never Land Books
The Never Land Books or Never Land Adventures are a series of short chapter books set in Never Land, the home of Peter Pan.

 They are based on the situations and characters established in the novel "Peter and the Starcatchers" and its sequels.

 Like the novels, they are written by Pulitzer Prize-winning humorist Dave Barry and suspense novelist Ridley Pearson, and illustrated by Greg Call.

 Although five books were planned, only three were published, in 2006–2008.

 The stories focus on supporting characters from the novels, such as the Indians, mermaids, pirates, and Lost Boys.

 They include:
</item 10>
=== END ITEM id=44fb8daeb48b37a0 ===

=== ITEM id=9f9ad2e1be6176de ===
Gold sentences and facts:
Sentence 0:  It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.
  Fact 0.0: The work is the third of Gluck's so-called reform operas for Vienna.
  Fact 0.1: The work follows 'Alceste' and 'Orfeo ed Euridice' among Gluck's reform operas for Vienna.
  Fact 0.2: The work is the least often performed of the three reform operas for Vienna.
Sentence 1: Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of Orpheus, set to a libretto by Ranieri de' Calzabigi.
  Fact 1.0: Orfeo ed Euridice is an opera composed by Christoph Willibald Gluck.
  Fact 1.1: Orfeo ed Euridice is based on the myth of Orpheus.
  Fact 1.2: Orfeo ed Euridice is set to a libretto by Ranieri de' Calzabigi.
  Fact 1.3: The French version of Orfeo ed Euridice is titled Orphée et Eurydice.
  Fact 1.4: The English title of Orfeo ed Euridice is 'Orpheus and Eurydice'.
Sentence 2:  The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.
  Fact 2.0: The piece was first performed at the Burgtheater in Vienna on 5 October 1762.
  Fact 2.1: The first performance of the piece was in the presence of Empress Maria Theresa.

Retrieved items:
<item 1>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 1>

<item 2>
Paride ed Elena
Paride ed Elena (] ; "Paris and Helen") is an opera by Christoph Willibald Gluck.

 It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.

 Like its predecessors, the libretto was written by Ranieri de' Calzabigi.

 The opera tells the story of the events between the Judgment of Paris and the flight of Paris and Helen to Troy.

 It was premiered at the Burgtheater in Vienna on 3 November 1770.
</item 2>

<item 3>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 3>

<item 4>
Gottlieb Muffat
Gottlieb Muffat (April 1690 – 9 December 1770), son of Georg Muffat, served as "Hofscholar" under Johann Fux in Vienna from 1711 and was appointed to the
position of third court organist at the "Hofkapelle" in 1717.

 He acquired additional duties over time including the instruction of members of the Imperial family, among them the future Empress Maria Theresa.

 He was promoted to second organist in 1729 and first organist upon the accession of Maria Theresa to the throne in 1741.

 He retired from official duties at the court in 1763.
</item 4>

<item 5>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 5>

<item 6>
Operetta (film)
Operetta (German: Operette) is a 1940 musical film directed by Willi Forst and starring Forst, Maria Holst and Dora Komar.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It is the first film in director Willi Forst's "Viennese Trilogy" followed by "Vienna Blood" (1942) and "Viennese Girls" (1945).

 The film portrays the life of Franz Jauner (1832–1900), a leading musical figure in the city.

 It is both an operetta film and a Wiener Film.
</item 6>

<item 7>
Das geheime Königreich
Das geheime Königreich ("The Secret Kingdom") is an opera in one act with words and music by Ernst Krenek, his Op.

 50 and the second of three one-acters (with "Der Diktator" and "Schwergewicht, oder Die Ehre der Nation") which premiered on 6 May 1928 at the Hessisches
Staatstheater Wiesbaden as part of the Mai-Festwoche Wiesbaden.

 It is subtitled "Märchenoper in 1 Akt " (fairy-tale opera in one act) and has been called a satirical fairy-tale opera.
</item 7>

<item 8>
Agrippina (opera)
Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.

 Composed for the 1709–10 Venice "Carnevale" season, the opera tells the story of Agrippina, the mother of Nero, as she plots the downfall of the Roman Emperor
Claudius and the installation of her son as emperor.

 Grimani's libretto, considered one of the best that Handel set, is an "anti-heroic satirical comedy", full of topical political allusions.

 Some analysts believe that it reflects Grimani's political and diplomatic rivalry with Pope Clement XI.
</item 8>

<item 9>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 9>

<item 10>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 10>
=== END ITEM id=9f9ad2e1be6176de ===

=== ITEM id=84202fac838d1529 ===
Gold sentences and facts:
Sentence 0: Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
  Fact 0.0: Distortion Mirrors is an English alternative rock band.
  Fact 0.1: Distortion Mirrors is based in Leeds, West Yorkshire, England.
  Fact 0.2: Distortion Mirrors was formed in 2011.
Sentence 1: OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.
  Fact 1.0: OK Go is an American rock band.
  Fact 1.1: OK Go was originally from Chicago, Illinois.
  Fact 1.2: OK Go is now based in Los Angeles, California.

Retrieved items:
<item 1>
OK Go videography
The musical rock band OK Go has earned considerable fame for their creative but often low-budget music videos, most of which have been promoted through Internet
video sharing sites like YouTube.

 Many of these have become viral videos; the 2006 video for "Here It Goes Again", in which the band performed a complex routine with the aid of motorized
treadmills, has received over 50 million views four years later.

 Their video for Needing/Getting, released February 5, 2012 in partnership with Chevrolet, debuted during Super Bowl XLVI and has over 32 million views on
YouTube.

 Samuel Bayer, who produced many music videos in the 1990s, asserted that OK Go's promotion of music videos on the Internet was akin to Nirvana's ushering in
the grunge movement.

 Many of the videos also use long or single-shot takes, which "Salon"' s Matt Zoller Seitz says "restore[s] a sense of wonder to the musical number by letting
the performers' humanity shine through and allowing them to do their thing with a minimum of filmmaking interference".
</item 1>

<item 2>
OK Go (2001 EP)
OK Go or The Pink EP (officially known as OKGoCD.002) is the second EP by American rock band OK Go.
</item 2>

<item 3>
Twelve Days of OK Go
Twelve Days of OK Go is a compilation album by American rock band OK Go.

 It was released on December 31, 2012.

 OK Go started releasing the songs on December 10, with one song released each weekday.

 The last song, a cover of "Any Time at All", was released on Christmas.

 A bonus track, a cover of "This Will Be Our Year," was released on New Year's Eve.
</item 3>

<item 4>
Distortion Mirrors
Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
</item 4>

<item 5>
OK Go (album)
OK Go is the debut studio album by American rock band OK Go.

 It was released in September 2002.

 The cover was created by designer Stefan Sagmeister.
</item 5>

<item 6>
OK Go (2000 EP)
OK Go or The Brown EP (officially known as OKGoCD.001) is the first EP by American rock band OK Go.
</item 6>

<item 7>
OK Go’s 2012 compilation *Twelve Days of OK Go* released one song each weekday from December 10, ending with “Any Time at All” on Christmas, then adding “This
Will Be Our Year” on New Year’s Eve. The Chicago-founded, Los Angeles-based band—Damian Kulash, Tim Nordwind, Dan Konopka, and Andy Ross—is known for elaborate,
often one-take viral videos and award-winning music clips. The band also released early EP
</item 7>

<item 8>
OK Go videography
The success of OK Go's music first won the band the 14th Annual Webby Special Achievement Award for Film and Video Artist of the Year.

 The video for "This Too Shall Pass" was named both "Video of the Year" and "Best Rock Video" at the 3rd annual UK Music Video Awards.

"This Too Shall Pass" won the LA Film Fest's Audience Award for Best Music Video, UK MVA Awards – Music Video of the Year Winner 2010, among others.
</item 8>

<item 9>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 9>

<item 10>
The Writing's on the Wall (OK Go song)
"The Writing's on the Wall" is a song by American rock band OK Go.

 It was released on June 17, 2014, as part of the band's EP "Upside Out", and is also the first single from the band's fourth studio album "Hungry Ghosts".

 On the same day, the band released a music video in which the members use props to create optical illusions, reflecting the song's description of a
relationship that fails because the couple has different points of view.

 Like previous OK Go videos, it is structured as a one-shot music video.

 The many YouTube views of the video caused the song to debut in the top ten of the US "Billboard" Hot Rock Songs chart, as well as number one on the Bubbling
Under Hot 100 Singles chart.
</item 10>
=== END ITEM id=84202fac838d1529 ===
