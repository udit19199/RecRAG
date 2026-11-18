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

=== ITEM id=fbd03bb6f2c9e412 ===
Gold sentences and facts:
Sentence 0: 3:10 to Yuma is a 2007 American Western film directed by James Mangold and produced by Cathy Konrad, and starring Russell Crowe and Christian Bale in the lead roles, with supporting performances by Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman.
  Fact 0.0: 3:10 to Yuma is a 2007 American Western film.
  Fact 0.1: 3:10 to Yuma was directed by James Mangold.
  Fact 0.2: 3:10 to Yuma was produced by Cathy Konrad.
  Fact 0.3: Russell Crowe and Christian Bale star in the lead roles of 3:10 to Yuma.
  Fact 0.4: Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman have supporting performances in 3:10 to Yuma.
Sentence 1: Alan Wray Tudyk ( ; born March 16, 1971) is an American actor and voice actor known for his roles as Hoban "Wash" Washburne in the space western television series "Firefly" and movie "Serenity", Alpha in the science fiction TV series "Dollhouse", Tucker McGee in "Tucker & Dale vs. Evil", Steve the Pirate in "", and Wat in "A Knight's Tale".
  Fact 1.0: Alan Wray Tudyk was born on March 16, 1971.
  Fact 1.1: Alan Tudyk is an American actor.
  Fact 1.2: Alan Tudyk is an American voice actor.
  Fact 1.3: Alan Tudyk is known for playing Hoban "Wash" Washburne in the space western television series "Firefly" and in the movie "Serenity".
  Fact 1.4: Alan Tudyk is known for playing Alpha in the science fiction TV series "Dollhouse".
  Fact 1.5: Alan Tudyk is known for playing Tucker McGee in "Tucker & Dale vs. Evil".
  Fact 1.6: Alan Tudyk is known for playing Steve the Pirate in "" (the title is missing from the source sentence).
  Fact 1.7: Alan Tudyk is known for playing Wat in "A Knight's Tale".

Retrieved items:
<item 1>
Henry Murdoch
Henry Murdoch (17 September, 1920 - 24 April, 1987), born as George Henry Murdock, was an Australian aboriginal actor and stockman who appeared in Australian
films of the 1940s and 1950s.

 He was working as stockman in Rockhampton when discovered by Ralph Smart, who was helping make "The Overlanders" (1946).

 The film's director, Harry Watt, later claimed Murdoch and fellow aboriginal actor Clyde Combo "proved to be first-class actors and were exceedingly quick
witted and intelligent.

 They certainly disproved the conventional idea that the Australian aboriginal is an animalistic caveman."
</item 1>

<item 2>
3:10 to Yuma (2007 film)
3:10 to Yuma is a 2007 American Western film directed by James Mangold and produced by Cathy Konrad, and starring Russell Crowe and Christian Bale in the lead
roles, with supporting performances by Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman.

 It is about a drought-impoverished rancher (Bale) who takes on the dangerous job of taking a notorious outlaw (Crowe) to justice.

 It is a remake of the , making it the second adaptation of Elmore Leonard's short story "Three-Ten to Yuma".

 Filming took place in various locations in New Mexico.

 "3:10 to Yuma" opened September 7, 2007, in the United States and received positive reviews from critics.
</item 2>

<item 3>
Gene Hackman
Eugene Allen "Gene" Hackman (born January 30, 1930) is a retired American actor and novelist.

 In a career spanning nearly five decades, Hackman was nominated for five Academy Awards, winning Best Actor in "The French Connection" and Best Supporting
Actor in "Unforgiven".

 He won three Golden Globes and two BAFTAs.
</item 3>

<item 4>
Alan Tudyk
Alan Wray Tudyk ( ; born March 16, 1971) is an American actor and voice actor known for his roles as Hoban "Wash" Washburne in the space western television
series "Firefly" and movie "Serenity", Alpha in the science fiction TV series "Dollhouse", Tucker McGee in "Tucker & Dale vs. Evil", Steve the Pirate in "", and
Wat in "A Knight's Tale".

 He also co-starred on the ABC sitcom "Suburgatory".

 Tudyk had a supporting role in the film "42" as baseball manager Ben Chapman and voiced roles in every Walt Disney Animation Studios film to date starting with
"Wreck-It Ralph".

 He has also featured in the web series "Con Man" and the films "" and "Rogue One: A Star Wars Story".
</item 4>

<item 5>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 5>

<item 6>
Deadfall (2012 film)
Deadfall is a 2012 American crime drama film directed by Stefan Ruzowitzky, written by Zach Dean and starring Eric Bana, Olivia Wilde, and Charlie Hunnam.
</item 6>

<item 7>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 7>

<item 8>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 8>

<item 9>
The Appaloosa
The Appaloosa (also known as Southwest to Sonora) is a 1966 American Western film Technicolor (set in the 1870s) from Universal Pictures starring Marlon Brando,
Anjanette Comer and John Saxon, who was nominated for a Golden Globe for Best Supporting Actor for his portrayal of a Mexican bandit.

 The film was directed by Sidney J. Furie, and shot in Wrightwood, Antelope Valley, and Lake Los Angeles, California, St. George, Utah, and Colorado City,
Arizona.
</item 9>

<item 10>
James Garner filmography
The following is the filmography for American actor James Garner.

 Garner was known for prominent roles in films such as "Sayonara" (1957) with Marlon Brando, "Cash McCall" (1960) with Natalie Wood, "The Great Escape" (1963)
with Steve McQueen, "The Thrill of It All" (1963) with Doris Day, "Move Over, Darling" (1963) with Doris Day, "The Wheeler Dealers" (1963) with Lee Remick,
Paddy Chayevsky's "The Americanization of Emily" (1964) with Julie Andrews, "Hour of the Gun" (1967), "Marlowe" (1969) with Bruce Lee, "Support Your Local
Sheriff!

"
</item 10>
=== END ITEM id=fbd03bb6f2c9e412 ===

=== ITEM id=fd03e4b028a9985e ===
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
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 1>

<item 2>
The Chinese Agent
The Chinese Agent (1970) is a comic novel by Michael Moorcock.

 It is a revision of "Somewhere in the Night", which Moorcock published in 1966 under the pseudonym Bill Barclay.

 Although Moorcock is best known as the author of fantasy fiction and science fiction-based parables such as "Behold the Man" and "The Dancers at the End of
Time", here he writes a light-hearted caper that parodies the spy novel genre.
</item 2>

<item 3>
A Cure for Cancer
A Cure for Cancer is a novel by British fantasy and science fiction writer Michael Moorcock, first published in London 1971 by Allison and Busby.

 The book is part of Moorcock's long-running Jerry Cornelius series.
</item 3>

<item 4>
Behold the Man (novel)
Behold the Man (1969) is a science fiction novel by British writer Michael Moorcock.

 It originally appeared as a novella in a 1966 issue of "New Worlds"; later, Moorcock produced an expanded version which was first published in 1969 by Allison
& Busby.

 The title derives from the Gospel of John, Chapter 19, Verse 5: "Then Jesus came out, wearing the crown of thorns and the purple robe.

 And Pilate said to them Behold the Man."
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
Warlord of the Air
The Warlord of the Air is a 1971 British alternate history science fiction novel written by Michael Moorcock.

 It concerns the adventures of Oswald Bastable, an Edwardian era soldier stationed in India, and his adventures in an alternate universe, in his own future,
wherein the First World War never happened.

 It is the first part of Moorcock's 'A Nomad of the Time Streams' trilogy and, in its use of speculative technology (such as airships) juxtaposed against an
Edwardian setting, it is widely considered to be one of the first steampunk novels.

 The novel was first published by Ace Books as part of their Ace Science Fiction Specials series.
</item 6>

<item 7>
The Final Programme
The Final Programme is a novel by British science fiction and fantasy writer Michael Moorcock.

 Written in 1965 as the underground culture was beginning to emerge, it was not published for several years.

 Moorcock has stated that publishers at the time considered it was "too freaky".
</item 7>

<item 8>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 8>

<item 9>
Michael Moorcock
Michael John Moorcock (born 18 December 1939) is an English writer, primarily of science fiction and fantasy, who has also published literary novels.

 He is best known for his novels about the character Elric of Melniboné, a seminal influence on the field of fantasy in the 1960s and 1970s.
</item 9>

<item 10>
Epic Pooh
"Epic Pooh" is a 1978 article by the British science fiction writer Michael Moorcock, which reviews the field of epic fantasy, with a particular focus on epic
fantasy written for children.

 In it Moorcock critiques J. R. R. Tolkien's "The Lord of the Rings" for its politically conservative assumptions and for being escapist literature.
</item 10>
=== END ITEM id=fd03e4b028a9985e ===

=== ITEM id=10b732880ee8514e ===
Gold sentences and facts:
Sentence 0:  The daughter of the American Consul in London, she was the first First Lady to be born outside the United States, or outside areas that were later to become part of the United States, such as the Thirteen Colonies – a distinction that would not be replicated until 192 years later by Melania Trump.
  Fact 0.0: She was the daughter of the American Consul in London.
  Fact 0.1: She was the first First Lady to be born outside the United States.
  Fact 0.2: She was the first First Lady to be born outside areas that were later to become part of the United States.
  Fact 0.3: Such areas include the Thirteen Colonies.
  Fact 0.4: The distinction of being the first First Lady born outside these areas was not replicated for 192 years.
  Fact 0.5: Melania Trump replicated this distinction 192 years later.
Sentence 1:  Before marrying future President Donald Trump in 2005, she worked as a fashion model; by 2016 she considered herself a "full-time mom."
  Fact 1.0: Before marrying future President Donald Trump in 2005, she worked as a fashion model.
  Fact 1.1: By 2016, she considered herself a 'full-time mom.'

Retrieved items:
<item 1>
Anna Harrison
Anna Tuthill Symmes Harrison (July 25, 1775 – February 25, 1864), wife of President William Henry Harrison and grandmother of President Benjamin Harrison, was
nominally First Lady of the United States during her husband's one-month term in 1841, but she never entered the White House.

 At the age of 65 years during her husband's presidential term, she is the oldest woman ever to become First Lady, as well as having the distinction of holding
the title for the shortest length of time, and the first person to be widowed while holding the title.

 She was the last First Lady to have been born in British America.
</item 1>

<item 2>
List of First Ladies of the United States
The First Lady of the United States is the hostess of the White House.

 The position is traditionally filled by the wife of the President of the United States, but, on occasion, the title has been applied to women who were not
presidents’ wives, such as when the president was a bachelor or widower, or when the wife of the president was unable to fulfill the duties of the First Lady
herself.

 The First Lady is not an elected position; it carries no official duties and receives no salary.

 Nonetheless, she attends many official ceremonies and functions of state either along with or in place of the president.

 Traditionally, the First Lady does not hold outside employment while occupying the office.

 She has her own staff, including the White House Social Secretary, the Chief of Staff, the Press Secretary, the Chief Floral Designer, and the Executive Chef.

 The Office of the First Lady is also in charge of all social and ceremonial events of the White House, and is a branch of the Executive Office of the
President.
</item 2>

<item 3>
Abigail Adams
Abigail Adams ("née" Smith; November 22  1744 – October 28, 1818) was the closest advisor and wife of John Adams, as well as the mother of John Quincy Adams.

 She is sometimes considered to have been a Founder of the United States, and is now designated as the first Second Lady and second First Lady of the United
States, although these titles were not in use at the time.
</item 3>

<item 4>
Louisa Adams
Louisa Catherine Johnson Adams (February 12, 1775 – May 15, 1852), wife of John Quincy Adams, was First Lady of the United States from 1825 to 1829.

 The daughter of the American Consul in London, she was the first First Lady to be born outside the United States, or outside areas that were later to become
part of the United States, such as the Thirteen Colonies – a distinction that would not be replicated until 192 years later by Melania Trump.
</item 4>

<item 5>
Melania Trump (born Melanija Knavs, later Knauss) is the current First Lady, a former model who married Donald Trump in 2005 and called herself a full-time mom
in 2016. First spouses include presidents’ and governors’ spouses; the unpaid, unofficial First Lady role has White House staff and ceremonial duties. Louisa
Adams was the first foreign-born First Lady; Anna Harrison served briefly without entering the White House. Other entries cover Idaho’s Vicki R
</item 5>

<item 6>
Vicki Risch
Vicki Risch is the former First Lady of Idaho and the wife of U.S. Sen. Jim Risch, who served as Governor of Idaho in 2006.

 She became First Lady on May 26, 2006, when her husband succeeded former Gov. Dirk Kempthorne, who resigned to become United States Secretary of the Interior.

 Mrs. Risch succeeded former First Lady Patricia Kempthorne who had held the post for over seven years.

 Mrs. Risch served as First Lady until January 2007, as her husband did not seek a full term as governor, but rather was reelected to his old post as lieutenant
governor.
</item 6>

<item 7>
List of current United States first spouses
In the United States, the first spouse (first lady for women, first gentleman for men) is the term used to refer to the spouse of a chief executive - that is,
of the spouse of the President of the United States (the First Lady of the United States and the First Gentleman of the United States) and the spouses of the
governors of the 50 U.S. states and U.S. territories (Guam, Northern Mariana Islands, Puerto Rico, American Samoa, and the United States Virgin Islands) and the
spouse of the mayor of the District of Columbia.

 (The spouses of many mayors are often called the "first lady" or "first gentleman" of the city as well, and the use of the terms sometimes extends even to the
spouses of college presidents).
</item 7>

<item 8>
Melania Trump
Melania Trump ( ; born Melanija Knavs; ] , Germanized to Melania Knauss; April 26, 1970) is the current First Lady of the United States.

 Before marrying future President Donald Trump in 2005, she worked as a fashion model; by 2016 she considered herself a "full-time mom."
</item 8>

<item 9>
Loreta Janeta Velázquez
Loreta Janeta Velázquez (June 26, 1842 – 1923), was a Cuban-born woman who masqueraded as a male Confederate soldier during the American Civil War.

 After her soldier husband's accidental death, she enlisted in the Confederate States Army in 1861.

 She then fought at Bull Run, Ball's Bluff, and Fort Donelson, but was discharged when her gender was discovered while in New Orleans.

 Undeterred, she reenlisted and fought at Shiloh, until unmasked once more.

 She then became a Confederate spy, working in both male and female guises, and as a double agent also reporting to the U.S. Secret Service.

 She remarried three more times, being widowed in each instance.

 According to William C Davis, she died in January 1923 under the name Loretta J. Beard after many years away from the public eye in a public psychiatric
facility, St. Elizabeths Hospital.

 She spied on the Union for about 5 years.
</item 9>

<item 10>
Office of the First Lady of the United States
The Office of the First Lady of the United States is the staff accountable to the First Lady of the United States.

 The office and its responsibilities, while not mandated, have grown as the role of the First Lady has grown and formalized through the history of the United
States.

 The Office of the First Lady is an entity of the White House Office, part of the Executive Office of the President.
</item 10>
=== END ITEM id=10b732880ee8514e ===

=== ITEM id=4621de15732e99bc ===
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
Tweed Volcano
Tweed Volcano is a partially eroded Early Miocene shield volcano located in northeastern New South Wales, which formed when this region of Australia passed over
the East Australia hotspot around 23 million years ago.

 Mount Warning, Lamington Plateau and the Border Ranges between New South Wales and Queensland are among the remnants of this volcano that was originally over
100 km in diameter and nearly twice the height of Mount Warning today, at 1156 m .

 Despite its size, Tweed Volcano was not a supervolcano; other shield volcanoes - such as on Hawaii - are much larger.

 In the 23 million years since the volcano was active, erosion has been extensive, forming a large erosion caldera around the volcanic plug of Mount Warning.

 Its erosion caldera is the largest in the Southern Hemisphere.
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
Two-liter bottle
The two-liter bottle is a common container for soft drinks.

 These bottles are produced from polyethylene terephthalate, also known as PET plastic, using the blow molding process.

 Bottle labels consist of a printed, tight-fitted plastic sleeve.

 A resealable screw-top allows the contents to be used at various times while retaining carbonation.
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
=== END ITEM id=4621de15732e99bc ===

=== ITEM id=2b093491b3bd164e ===
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
Diocese of Salisbury
The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.

 The diocese covers most of Dorset (excepting the deaneries of Bournemouth and Christchurch, which fall within the Diocese of Winchester), and most of Wiltshire
(excepting a part in the north and Swindon).

 The diocese is led by the Bishop of Salisbury (Nick Holtam) and the diocesan synod.

 The bishop's seat is at Salisbury Cathedral.
</item 1>

<item 2>
Lincoln Cathedral
Lincoln Cathedral or the Cathedral Church of the Blessed Virgin Mary of Lincoln, and sometimes St. Mary's Cathedral in Lincoln, England is the seat of the
Anglican bishop.

 Building commenced in 1088 and continued in several phases throughout the medieval period.

 It was the tallest building in the world for 238 years (1311–1549), and the first building to hold that title after the Great Pyramid of Giza.

 The central spire collapsed in 1549 and was not rebuilt.

 The cathedral is the third largest in Britain (in floor area) after St Paul's and York Minster, being 484 by .

 It is highly regarded by architectural scholars; the eminent Victorian writer John Ruskin declared: "I have always held... that the cathedral of Lincoln is out
and out the most precious piece of architecture in the British Isles and roughly speaking worth any two other cathedrals we have."
</item 2>

<item 3>
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
</item 3>

<item 4>
St Philip's Cathedral, Birmingham
The Cathedral Church of Saint Philip is the Church of England cathedral and the seat of the Bishop of Birmingham.

 Built as a parish church and consecrated in 1715, St Philip's became the cathedral of the newly formed Diocese of Birmingham in 1905.

 St Philip's was built in the early 18th century in the Baroque style by Thomas Archer and is located on Colmore Row, Birmingham, England.

 The cathedral is a Grade I listed building.

 St Philip's is the third smallest cathedral in England after Derby and Chelmsford.
</item 4>

<item 5>
Guildford Cathedral
The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.

 The Lord Onslow donated the land on which the Cathedral was built with a covenant for this purpose.

 Designed by Sir Edward Maufe and built between 1936 and 1961, it is the seat of the Diocese of Guildford.
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
Diocese of Guildford
The Diocese of Guildford is a Church of England diocese covering nine of the eleven districts in Surrey, much of north-east Hampshire and a parish in Greater
London.

 The cathedral is Guildford Cathedral and the bishop is the Bishop of Guildford.

 Of the two provinces of the church it falls within the Province of Canterbury.
</item 7>

<item 8>
Chelmsford Cathedral
Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.

 It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
</item 8>

<item 9>
St Paul's Cathedral, Rockhampton
St Paul's Anglican Cathedral is a heritage-listed cathedral at 89 William Street, Rockhampton, Rockhampton Region, Queensland, Australia.

 It was designed principally by Annersley Voysey and built from 1883 to 1953.

 It is also known as St Paul's Church of England and St Paul's Church of England Cathedral.

 It was added to the Queensland Heritage Register on 21 October 1992.

 The associated hall and offices were listed on the Queensland Heritage Register on 23 June 2000.
</item 9>

<item 10>
Exterior sculpture of Guildford Cathedral
The Exterior sculpture of Guildford Cathedral provides many artistic features, including sculptures, engravings and more by some of England’s finest sculptors
and craftsmen of the 1950s and 1960s.

 The people who worked on the cathedral include: Edward Maufe, Alan Collins, Vernon Hill, Eric Gill, John Hutton, Dennis Huntley and others.
</item 10>
=== END ITEM id=2b093491b3bd164e ===

=== ITEM id=724f137978c0e9be ===
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
Reef Hawaiian Pro
The Reef Hawaiian Pro is a surfing event held annually on the North Shore, Oahu, Hawaii.

 It is currently the first leg of the Triple Crown of Surfing.

 Surfers compete for a share of $135,000 (men) or $35,000 (women) prize money.

 Since it carries a 6-star rating, the event is critical for those who want to qualify for the ASP World Tour.
</item 1>

<item 2>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 2>

<item 3>
Glenwood Caverns Adventure Park
Glenwood Caverns Adventure Park is an adventure park located above Glenwood Springs, Colorado, about 160 miles west of Denver.

 Prior to 2003, only cave tours were available until a major expansion took place.

 The park is unique because it sits at an altitude of 7,100 feet on a mountain above Glenwood.

 Today, the park features numerous attractions in addition to the cave tours.
</item 3>

<item 4>
Treasure Island Resort &amp; Casino
An expansion that was completed in fall 2008 included 230 new hotel rooms, 30,000-square-foot event center and a bowling center complete with an arcade area.

 In 2015, Tado Steakhouse was constructed, Tradewinds Buffet was remodeled and the water park & spa construction began.

 The Lagoon and Wave Spa opened February 9, 2016.
</item 4>

<item 5>
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
</item 5>

<item 6>
Christchurch Adventure Park
Christchurch Adventure Park is an Adventure park in the Port Hills of Christchurch, New Zealand.

 Built by the Canadian company Select Evolution, it had its opening function on 16 December 2016.

 The 1500 holders of special passes could ride from 17 December, while the park opened to the public on 21 December.

 According to the developer, the 358 ha park with a 1.8 km chairlift and initially 50 km of downhill tracks is the largest facility of its kind in the Southern
Hemisphere.

 The park closed on 13 February as a precaution due to a nearby fire and two days later, most of the park's tree cover had been destroyed in a large wildfire.

 The lift and ziplines have suffered significant damage, and the park's condition is much worse than the operator initially thought.

 No reopening date has been set, and "it won't be a quick fix".
</item 6>

<item 7>
Wild West Express Coaster
Wild West Express Coaster is a steel roller coaster operating at Glenwood Caverns Adventure Park in Glenwood Springs, Colorado.

 Wild West Express Coaster opened to the public at Glenwood Caverns Adventure Park on May 25, 2012.
</item 7>

<item 8>
White Water Landing (Dorney Park)
White Water Landing is a Shoot-the-Chutes ride at Dorney Park & Wildwater Kingdom amusement park in Allentown, Pennsylvania.

 It was built in 1992, 10 years after a sister park named Cedar Point Built a defunct flume using the same name: White Water Landing (Cedar Point).

 This ride is identical to Snake River Falls at Cedar Point in Sandusky, Ohio.

 It is located in the upper side near Hydra the Revenge.
</item 8>

<item 9>
Luna Park, Schenectady
Roughly seven decades before the Skycoaster rides that now dot various United States amusement parks, Luna/Rexford Park featured an aerial swing ride.
</item 9>

<item 10>
Pirate Adventure Park
Pirate Adventure Park is a small theme park located beside Westport House in Westport, Ireland.

 The Park is home to the first flume ride in Ireland (Pirates Plunge).

 The Park is Pirate themed due to Wesport House's association with Grace O'Malley.

 The Park was developed by The 11th Marquess of Sligo on the grounds of Westport House.
</item 10>
=== END ITEM id=724f137978c0e9be ===

=== ITEM id=e170347c04ef58d9 ===
Gold sentences and facts:
Sentence 0:  He has acted in a few Off Broadway productions and has made appearances on American television shows such as "", "Person of Interest" and "" and video games such as "Grand Theft Auto V", "Red Dead Redemption", "" and "Red Dead Revolver".
  Fact 0.0: This person has acted in a few Off Broadway productions.
  Fact 0.1: This person has made appearances on American television shows such as Person of Interest.
  Fact 0.2: This person has made appearances on an American television show whose title is blank in the sentence.
  Fact 0.3: This person has made appearances in video games such as Grand Theft Auto V.
  Fact 0.4: This person has made appearances in video games such as Red Dead Redemption.
  Fact 0.5: This person has made appearances in video games such as Red Dead Revolver.
  Fact 0.6: This person has made appearances in a video game whose title is blank in the sentence.
Sentence 1: Red Dead Redemption is a Western action-adventure video game developed by Rockstar San Diego and published by Rockstar Games.
  Fact 1.0: Red Dead Redemption is a Western action-adventure video game.
  Fact 1.1: Red Dead Redemption was developed by Rockstar San Diego.
  Fact 1.2: Red Dead Redemption was published by Rockstar Games.

Retrieved items:
<item 1>
Red Dead Revolver
Red Dead Revolver is a 2004 western action-adventure third-person shooter video game developed by Rockstar San Diego, published by Rockstar Games and
distributed by Take-Two Interactive for the PlayStation 2 and Xbox.

 It is the first game in the "Red Dead" series.
</item 1>

<item 2>
Red Dead Redemption
Red Dead Redemption is a Western action-adventure video game developed by Rockstar San Diego and published by Rockstar Games.

 It was released for PlayStation 3 and Xbox 360 consoles in May 2010.

 It is the second title in the "Red Dead" franchise, after 2004's "Red Dead Revolver".

 The game, set during the decline of the American Frontier in the year 1911, follows John Marston, a former outlaw whose wife and son are taken hostage by the
government in ransom for his services as a hired gun.

 Having no other choice, Marston sets out to bring the three members of his former gang to justice.
</item 2>

<item 3>
Red Dead
Red Dead is a series of western action-adventure video games developed by Rockstar San Diego and published by Rockstar Games.

 The first game, titled "Red Dead Revolver", was originally released for the PlayStation 2 and Xbox in May 2004.

 The second game, "Red Dead Redemption", was released for the PlayStation 3 and Xbox 360 in May 2010 to wide critical acclaim, and is considered one of the
greatest games of all time.

 The third entry in the series, "Red Dead Redemption II", is scheduled for release in Q2 2018 for PlayStation 4 and Xbox One.
</item 3>

<item 4>
List of Grand Theft Auto V characters
Rockstar North's 2013 open world action-adventure video game "Grand Theft Auto V", published by Rockstar Games, deals with a trio of criminals and their efforts
to commit heists while under pressure from a corrupt government agency.

 The game's use of three lead protagonists is a break from series tradition.

 Three days after its release, "Grand Theft Auto V" had earned more than US $1 billion in sales, making it the fastest selling video game in history.

 Additionally, "Grand Theft Auto V" was voted Game of the Year from several gaming publications.

 The game's characters were created by writers Dan Houser and Rupert Humphries.

 The use of multiple lead protagonists, as well as the cast in general, has received generally positive reviews from gaming magazines and websites.
</item 4>

<item 5>
Tyler Bunch
Tyler T. Bunch (born February 10, 1970) is an American puppeteer, puppet designer, director, and actor.

 In addition to his work for the Jim Henson Company, Bunch has created and performed puppets for PBS, Nickelodeon and Disney's The Muppets Studio.

 He has acted in a few Off Broadway productions and has made appearances on American television shows such as "", "Person of Interest" and "" and video games
such as "Grand Theft Auto V", "Red Dead Redemption", "" and "Red Dead Revolver".

 Currently, he is directing the PBS children's television series "SeeMore's Playhouse".

 He is also currently scheduled to be one of the puppeteers for the musical stage adaptation of "Emmet Otter's Jug-Band Christmas".

 He resides in Stratford, Connecticut.
</item 5>

<item 6>
Controversies surrounding Grand Theft Auto V
"Grand Theft Auto V" is an open world, action-adventure video game developed by Rockstar North and published by Rockstar Games.

 Upon its 17 September 2013 release on PlayStation 3 and Xbox 360, the game generated controversies related to its violence and depiction of women.

 A mission that requires players to use torture equipment in a hostage interrogation polarised reviewers, who noted its political commentary but felt that the
torture sequence was in poor taste.

 The mission also received criticism from politicians and anti-torture charity groups.

 The game became subject to widespread online debate over its portrayal of women, particularly in the wake of backlash against GameSpot journalist Carolyn Petit
after she claimed the game was misogynystic in her review.

 After Petit's review webpage received more than 20,000 largely negative comments, many journalists defended her right to an opinion and lamented the gaming
community's defensiveness towards criticism.

 Television personality Karen Gravano and actress Lindsay Lohan both filed lawsuits against Rockstar in allegation that characters in the game were based on
their likenesses.
</item 6>

<item 7>
Grand Theft Auto: Vice City Stories
Grand Theft Auto: Vice City Stories is an open world action-adventure video game developed by Rockstar Leeds in conjunction with Rockstar North and published by
Rockstar Games.

 Released on 31 October 2006 for the PlayStation Portable and later for the PlayStation 2 on 5 March 2007.

 The game is the tenth instalment in the "Grand Theft Auto" series and the sixth and last game in the 3D universe.

 This was also the last game in the series to use well-known and Hollywood voice talent; "Grand Theft Auto IV" onward would instead use lesser-known and unknown
voice actors.

 It is a prequel to "" and successor to the previous PSP release, "".

 Vice City is a representation of Miami, Florida.
</item 7>

<item 8>
List of accolades received by Grand Theft Auto V
"Grand Theft Auto V" is an open world, action-adventure video game developed by Rockstar North and published by Rockstar Games.

 Players freely roam the fictional state of San Andreas (based on Southern California), composed of open countryside and the fictional city of Los Santos (based
on Los Angeles).

 The game was first announced on 25 October 2011, and was widely anticipated.

 At the 2012 Spike Video Game Awards (now the VGX), it was awarded Most Anticipated Game.

 It was released on 17 September 2013 for the PlayStation 3 and Xbox 360, on 18 November 2014 for the PlayStation 4 and Xbox One, and on 14 April 2015 for
Microsoft Windows.

 Review aggregator, Metacritic, assigned the game a normalised score of 97 out of 100, indicating "universal acclaim", based on 50 reviews for the PlayStation 3
version, 58 reviews for the Xbox 360 version, 66 reviews for the PlayStation 4 version and 14 reviews for the Xbox One version.
</item 8>

<item 9>
Red Dead Redemption II
Red Dead Redemption II is an upcoming western action-adventure video game developed and published by Rockstar Games for release on PlayStation 4 and Xbox One in
Q2 2018.

 The game is a prequel to the 2010 title "Red Dead Redemption" and will be the third entry in the "Red Dead" series.

 It follows outlaw Arthur Morgan, a member of the Dutch van der Linde gang.
</item 9>

<item 10>
Development of Grand Theft Auto V
An approximately 1,000-person team developed "Grand Theft Auto V", an action-adventure video game, over several years.

 Rockstar Games released "Grand Theft Auto V" on 17 September 2013 for PlayStation 3 and Xbox 360, on 18 November 2014 for PlayStation 4 and Xbox One, and on 14
April 2015 for Microsoft Windows, as the fifteenth entry in the "Grand Theft Auto" series.

 Development began soon after "Grand Theft Auto IV"' s release and was led by Rockstar North's core 360-person team, who collaborated with several other
international Rockstar Games studios.

 The team considered the game a spiritual successor to many of their previous projects like "Red Dead Redemption" and "Max Payne 3".

 After its unexpected announcement in 2011, the game was fervently promoted with press showings, cinematic trailers, viral marketing strategies and special
editions. Its release date, though subject to several delays, was widely anticipated.
</item 10>
=== END ITEM id=e170347c04ef58d9 ===

=== ITEM id=ec6bfbad6a81ee60 ===
Gold sentences and facts:
Sentence 0: Ghost Rider is a song by the protopunk band Suicide appearing on their debut album.
  Fact 0.0: Ghost Rider is a song.
  Fact 0.1: Ghost Rider is a song by the protopunk band Suicide.
  Fact 0.2: Ghost Rider appears on the debut album of the protopunk band Suicide.
Sentence 1:  The song is based on the Marvel Comics character.
  Fact 1.0: The song Ghost Rider is based on the Marvel Comics character.
Sentence 2: Suicide is the debut album from the American rock band Suicide.
  Fact 2.0: The album Suicide is the debut album of the American rock band Suicide.
  Fact 2.1: The band Suicide is an American rock band.
Sentence 3:  It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.
  Fact 3.0: The album Suicide was released in 1977.
  Fact 3.1: The album Suicide was released on Red Star Records.
  Fact 3.2: The album Suicide was produced by Craig Leon.
  Fact 3.3: The album Suicide was produced by Marty Thau.

Retrieved items:
<item 1>
A Boy in a Man's World
A Boy in a Man's World is the second studio album by Mucky Pup.

 The album was recorded at Nevessa Productions and released in 1989 through Torrid Records and Roadrunner Records.

 The album features a re-recorded version of "U-Stink-But-I-♥-U," which the band had previously won a Bloom County songwriting contest with.

 The band would also shoot their first music video for this song.

 The album also features "Batman", a song based on an urban legend that also contains a snippet of the title theme to the 1966 "Batman" TV.
</item 1>

<item 2>
Ghost Rider (Suicide song)
Ghost Rider is a song by the protopunk band Suicide appearing on their debut album.

 The song is based on the Marvel Comics character.
</item 2>

<item 3>
Muzzle (song)
"Muzzle" is a song by American alternative rock band The Smashing Pumpkins from their third album, "Mellon Collie and the Infinite Sadness".

 It was one of the last songs written by Billy Corgan for "Mellon Collie", with the song's lyrics referring to what Corgan thought the public's perception was
of him at the time.

 It was rumored to be the Smashing Pumpkins fifth and final single from this album, as is evidenced by the fact that a promotional single for the song was
issued to radio stations worldwide.

 However, the song "Thirty-Three" was released as the fifth and final single instead.
</item 3>

<item 4>
Live at Winterland 1978 (Avengers album)
Live at Winterland 1978 is a live album by the Avengers.

 It was released through online music stores on February 16, 2010.

 The album features a recording of their set on January 14, 1978 at the Winterland Ballroom in San Francisco, CA, when they opened for the Sex Pistols on what
would become the Sex Pistols final show, before reuniting years later.

 The Sex Pistols' set has been released on an album of the same name.

 Songs from this album were previously released on a couple 7" bootlegs titled "Penelope" and "Summer of Hate".
</item 4>

<item 5>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 5>

<item 6>
The Aeroplane Flies High
The Aeroplane Flies High is a five-disc box set released by American alternative rock band The Smashing Pumpkins in 1996.

 It contains expanded versions of the five singles from their album "Mellon Collie and the Infinite Sadness" and also included a 44-page booklet with pictures
and writings by the band's lead singer Billy Corgan, as well as lyrics.

 A limited edition release, the box reached number 42 on the "Billboard" charts, and sold 300,000 units (1.5 million discs in all), generating a platinum disc
for the band.

 Originally intended to be limited to 200,000 copies, Virgin Records produced more after the original run sold out due to overwhelming and unexpected demand.

 The album was remastered in 2013 under the supervision of frontman Billy Corgan and reissued on vinyl and as a CD/DVD box set.
</item 6>

<item 7>
It's Superman!
It's Superman!

 is a novel by Tom De Haven based on the comic book superhero Superman.

 It was released on September 15, 2005 in hardcover and August 29, 2006 in paperback.

 The premise tells the tale of Clark Kent's beginnings into becoming Superman, set in the 1930s, where Clark befriends a wrongly convicted photographer named
Willi Berg, and is then taken from Kansas to Hollywood and finally in New York where he meets Lois Lane, fights Lex Luthor, as he debuts in his superhero
persona.

 Despite the setting, this is not about the Golden Age Superman also known as the Superman of Earth-2; as Perry White, the Daily Planet, Lex Luthor's position
and his trademark powers (including flight) are not part of that alternate Earth.

 Rather, it's a Superman period piece set in the 1930s.
</item 7>

<item 8>
Superman: A Tale of Five Cities
Superman og Fredsbomben (translation: Superman and the Peace Bomb) or Superman: A Tale of Five Cities is a large prestige format 48-page graphic novel published
by Danish publisher Interpresse in 1990.

 In celebration of Superman's 50th anniversary in 1988, DC Comics incentived overseas publishers to produce an original story with the Man of Steel.

 Only the Interpresse editors from Denmark attended the call, comic book translator and writer Niels Søndergaard wrote the story featuring Clark Kent and Lois
Lane in a cold war adventure where Superman needs to devise a Lex Luthor plan to control all Nuclear Bombs in Europe.

 The storyline presents the Man of Steel visiting five capitals from northern Europe and Scandinavia: Amsterdam, Copenhagen, Oslo, Stockholm and Helsinki.

 The art and colors are provided by Teddy Kristiansen and the lettering by Rebecca Løwe.

 It was dedicated to danish editors Henning Kure and Ove Høyer, pioneers of the super-heroes comics in Denmark.
</item 8>

<item 9>
List of The Simpsons comics
The following is a list of comic book series based on the animated TV show The Simpsons and published by Bongo Comics in the United States.

 The first comic strips based on "The Simpsons" appeared in 1991 in the magazine "Simpsons Illustrated" (not to be confused with the comic publications from
2012 bearing the same name), which was a companion magazine to the show.

 The comic strips were popular and a one-shot comic book entitled "Simpsons Comics and Stories", containing three different stories, was released in 1993 for
the fans.

 The book was a success and due to this, the creator of "The Simpsons", Matt Groening, and his companions Bill Morrison, Mike Rote, Steve Vance and Cindy Vance
created the publishing company Bongo Comics.

 By the end of 1993, Bongo was publishing four titles: "Simpsons Comics", "Bartman", "Radioactive Man" and "Itchy & Scratchy Comics".
</item 9>

<item 10>
The Suicide File
The Suicide File was an American hardcore punk band from Boston, Massachusetts that formed in April 2001.

 The band wrote songs with a mostly political message, although many songs also dealt with social and personal problems.

 Most of the band's output was released on the Southern California-based hardcore label Indecision Records.

 The band reunited in June 2006 to embark on their first European Tour.

 Members of the bands are or have been affiliated with The Hope Conspiracy, Death By Stereo, When Tigers Fight, Adamantium, Give Up the Ghost/American
Nightmare, Clouds, Panic and many more.

 Most of these are bands with whom Alexander has drummed for short periods of time.

 Dave Weinberg was also known for his frequent collaborations and duets with Julie Ecker and James "Boom Boom" Auclair (cf. unpublished correspondence with the
artists).

 Despite their short tenure, The Suicide File continues to be revered within the hardcore punk community for their musical output and lasting impression on the
flourishing Boston hardcore scene.

 The band since 2006 has played a small number of successfully sized reunion sets.
</item 10>
=== END ITEM id=ec6bfbad6a81ee60 ===

=== ITEM id=cf75607de8e3b19f ===
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
The Brightlights
The Brightlights are a four-piece British indie rock band from Grimsby in North East Lincolnshire.

 They released their debut single, "Inspired By", on 5 November 2007.

 The single was produced by Feeder frontman Grant Nicholas.
</item 1>

<item 2>
Renegades (band)
Renegades were a British rock band (not to be confused with the sixties band "The Renegades"), which started out as a side-project from two members of the band
Feeder, featuring guitarist Grant Nicholas and bassist Taka Hirose, before becoming a pseudonym name for Feeder themselves.

 Nicholas formed Renegades alongside Hirose with Karl Brazil from Ben's Brother, who completed the group and a 4-track EP was then recorded.

 Soon later, Renegades became an alternative name for Feeder at various concerts where they would play an entire show pretending not to be Feeder, but a
different band with the same members.

 If Feeder songs released before the "Renegades" album were to be played, the band would announce that they're covering Feeder songs.

 Grant Nicholas once introduced "Tangerine" as "A cover of a song from a band we know".
</item 2>

<item 3>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 3>

<item 4>
Fayard Nicholas
Fayard Antonio Nicholas (October 20, 1914 – January 24, 2006) was an American choreographer, dancer and actor.

 He and his younger brother Harold Nicholas made up the Nicholas Brothers tap-dance duo, who starred in the MGM musicals "An All-Colored Vaudeville Show"
(1935), "Stormy Weather" (1943), "The Pirate" (1948), and "Hard Four" (2007).

 The Nicholas brothers also starred in the 20th Century-Fox musicals "Down Argentine Way" (1940), "Sun Valley Serenade" (1941), and "Orchestra Wives" (1942).
</item 4>

<item 5>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 5>

<item 6>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 6>

<item 7>
Raindancer
Raindancer was an electroacoustic rock band formed in Newport, Wales in the early 1990s.

 The band featured Grant Nicholas and Jon Lee, who later became founding members of Feeder.
</item 7>

<item 8>
Buck Rogers (song)
"Buck Rogers" is the eleventh single by Feeder.

 It was the first single to be taken from the "Echo Park" album and was released on The Echo Label.

 The track reached number 5 in the UK Singles Chart after its release on 8 January 2001.

 The group had originally not intended the track to be one of theirs, as frontman and main songwriter of the band Grant Nicholas, originally wrote "Buck Rogers"
for SR-71, only for producer Gil Norton and A&R staff of Echo to convince the band they could have a hit with it themselves, after hearing a demo recorded by
Feeder.

 It continues to be played on UK alternative radio stations as a classic hit of its genre during the early 2000s.
</item 8>

<item 9>
Yorktown Heights (album)
Yorktown Heights was the only solo album by Feeder frontman Grant Nicholas.

 The album was released on 11 August 2014 in the United Kingdom and recorded during 2013, being a year after Nicholas announced at Feeder's final show of 2012
at the Brixton Academy that the band would be taking a break.

 It was with the first airplay of ""Soul Mates"" on the BBC Radio 2 show "The Dermot O'Leary Show" on 31 May 2014, that Grant's solo career made its radio
debut.
</item 9>

<item 10>
Confederate Railroad
Confederate Railroad is an American country rock–Southern rock band founded in 1987 in Marietta, Georgia, by Danny Shirley (lead vocals, rhythm guitar), Michael
Lamb (lead guitar), Mark Dufresne (drums), Chris McDaniel (keyboards), Warren "Gates" Nichols (steel guitar) and Wayne Secrest (bass guitar).

 After serving as a backing band for outlaw country acts David Allan Coe and Johnny Paycheck, the band signed to a recording contract with Atlantic Records,
releasing their self-titled debut album that year.

 In the 1990s, they released four more albums for Atlantic.
</item 10>
=== END ITEM id=cf75607de8e3b19f ===

=== ITEM id=6072ac50a3172ba4 ===
Gold sentences and facts:
Sentence 0: Wheelock Whitney may refer to one of three members of the Whitney family:
  Fact 0.0: Wheelock Whitney may refer to one of three members of the Whitney family.
Sentence 1: The Whitney family is an American family notable for their social prominence, wealth, business enterprises and philanthropy, founded by John Whitney (1592–1673) who came from London, England to Watertown, Massachusetts in 1635.
  Fact 1.0: The Whitney family is an American family.
  Fact 1.1: The Whitney family is notable for their social prominence, wealth, business enterprises and philanthropy.
  Fact 1.2: The Whitney family was founded by John Whitney (1592–1673).
  Fact 1.3: John Whitney came from London, England.
  Fact 1.4: John Whitney came to Watertown, Massachusetts.
  Fact 1.5: John Whitney came to Watertown, Massachusetts in 1635.

Retrieved items:
<item 1>
Whitney family
The Whitney family is an American family notable for their social prominence, wealth, business enterprises and philanthropy, founded by John Whitney (1592–1673)
who came from London, England to Watertown, Massachusetts in 1635.

 The historic family mansion in Watertown, known as The Elms, was built in 1710.
</item 1>

<item 2>
James Scollay Whitney
James Scollay Whitney (May 19, 1811 – October 24, 1878) was an American business executive and politician.

 He was the father of Henry Melville Whitney and William Collins Whitney, founders of the Whitney family business interests.
</item 2>

<item 3>
Charles Andrew Whitney
Charles Andrew Whitney (November 14, 1864- December 31, 1912) was an American businessman and industrialist in the late 19th century, born in Princeton,
Massachusetts.

 He was part of the prominent American Whitney family.

 In 1859, Charles, his brother Levi L. Whitney, and Orville E. Thompson helped lead the large-scale manufacturing of leather boots and shoes in Chicago and were
attributed with successfully running the first factory of its kind there.

 Whitney himself held patents for the manufacture of leather.
</item 3>

<item 4>
Joseph Saunders led Visa; Gertrude Vanderbilt Whitney was a sculptor and founded the Whitney Museum. Roscoe Channing was a Princeton All-American, Rough Rider,
and mining executive. Fayez Sarofim is a Houston billionaire investor and Texans co-owner; Charles Agemian was a senior banker. The Rockefeller Brothers Fund,
founded in 1940, supports philanthropy. Bankers William Harrison and Walter Shipley led major JPMorgan predecessor mergers. Rockefeller family members include
philanthropists
</item 4>

<item 5>
Wheelock Whitney
Wheelock Whitney may refer to one of three members of the Whitney family:
</item 5>

<item 6>
Harry Payne Whitney
Harry Payne Whitney (April 29, 1872 – October 26, 1930) was an American businessman, thoroughbred horse breeder, and member of the prominent Whitney family.
</item 6>

<item 7>
Gertrude Vanderbilt Whitney
Gertrude Vanderbilt Whitney (January 9, 1875 – April 18, 1942) was an American sculptor, art patron and collector, and founder in 1931 of the Whitney Museum of
American Art in New York City.

 She was a prominent social figure and hostess, who was born into the wealthy Vanderbilt family and married into the Whitney family.
</item 7>

<item 8>
Rockefeller family
The Rockefeller family ( ) is an American industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the
late 19th and early 20th centuries, with John D. Rockefeller and his brother William Rockefeller primarily through Standard Oil.

 The family is also known for its long association with and control of Chase Manhattan Bank.

 They are considered to be one of the most powerful families, if not the most powerful family, in the history of the United States.
</item 8>

<item 9>
John Hay Whitney
John Hay Whitney (August 17, 1904 – February 8, 1982), colloquially known as Jock Whitney, was U.S. Ambassador to the United Kingdom, publisher of the "New York
Herald Tribune", and president of the Museum of Modern Art.

 He was a member of the Whitney family.
</item 9>

<item 10>
Ross Winans
Ross Winans (1796–1877) was an American inventor, mechanic, and builder of locomotives and railroad machinery.

 He is also noted for design of pioneering cigar-hulled ships.

 Winans, one of the United States' first multi-millionaires, was involved in national and state politics, a southern-sympathizer and was a vehement "states'
rights" advocate.

 His outspoken anti-federal stance as a member of the Maryland House of Delegates, the lower chamber of the General Assembly, (state legislature) led to his
temporary arrest on board a Baltimore and Ohio Railroad train returning from an early session of the legislature held in the western Maryland town of Frederick
to avoid the Union Army-occupied state capital of Annapolis in April–May, 1861, to consider the possibilities of state secession, during the early decisive
period of the American Civil War.

 Winans was related to James McNeill Whistler through marriage (Whistler's brother George married Winans' daughter Julia).
</item 10>
=== END ITEM id=6072ac50a3172ba4 ===

=== ITEM id=c98a2d88973b036b ===
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
The Picture of Dorian Gray (1917 film)
The Picture of Dorian Gray (German: Das Bildnis des Dorian Gray) is a 1917 German silent fantasy film directed by Richard Oswald and starring Bernd Aldor, Ernst
Pittschau, and Ernst Ludwig.

 The film is based on the novel "The Picture of Dorian Gray" by Oscar Wilde.
</item 1>

<item 2>
Dorian Gray (character)
Dorian Gray is the main character in Oscar Wilde's novel The Picture of Dorian Gray.

 Dorian Gray is a handsome, susceptible, wealthy young Victorian gentleman.

 He falls in love with and devotes himself to Sibyl Vane.

 Dorian has two friends, Basil and Henry.

 Basil is an artist who draws a portrait of Dorian that emphasizes his youthful, noble, and natural beauty.

 Henry is a friend who has a cynical nature and who shares various thoughts with Dorian.
</item 2>

<item 3>
The Picture of Dorian Gray (1945 film)
The Picture of Dorian Gray is a 1945 American horror-drama film based on Oscar Wilde's 1890 novel of the same name.

 Released in March 1945 by Metro-Goldwyn-Mayer, the film is directed by Albert Lewin and stars George Sanders as Lord Henry Wotton and Hurd Hatfield as Dorian
Gray.

 Shot primarily in black-and-white, the film features four inserts in 3-strip Technicolor of Dorian's portrait as a special effect (the first two of his
portrait's original state, and the second two after a major period of degeneration).
</item 3>

<item 4>
El retrato de Dorian Gray
The Picture of Dorian Gray ("El retrato de Dorian Gray"), is a Mexican telenovela, based on "The Picture of Dorian Gray", the novel written by Oscar Wilde.

 The main character is the handsome young man called Dorian Gray (played by late Enrique Álvarez Félix).

 This telenovela was made in 1969.
</item 4>

<item 5>
Dorian Gray (1970 film)
Dorian Gray (Italian: "Il dio chiamato Dorian") aka "The Sins of Dorian Gray" is a 1970 movie adaptation of Oscar Wilde's novel "The Picture of Dorian Gray"
starring Helmut Berger.

 The Italian title translates as "A God Called Dorian".
</item 5>

<item 6>
The Final Programme
The Final Programme is a novel by British science fiction and fantasy writer Michael Moorcock.

 Written in 1965 as the underground culture was beginning to emerge, it was not published for several years.

 Moorcock has stated that publishers at the time considered it was "too freaky".
</item 6>

<item 7>
A Portrait of Dorian Gray
A Portrait of Dorian Gray (2005) is the fashion designer and photographer Karl Lagerfeld's rendition of Oscar Wilde's novel "The Picture of Dorian Gray" (1891)
in photography.

 The models, Larry Scott and Eva Herzigová, star as Mr. and Mrs. Dorian Gray.
</item 7>

<item 8>
The Picture of Dorian Gray (1910 film)
The Picture of Dorian Gray (1910), a.k.a. Dorian Grays Portræt, is a Danish silent film based on the novel "The Picture of Dorian Gray" by Oscar Wilde.
</item 8>

<item 9>
The Picture of Dorian Gray
The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".

 The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.

 Despite that censorship, "The Picture of Dorian Gray" offended the moral sensibilities of British book reviewers, some of whom said that Oscar Wilde merited
prosecution for violating the laws guarding the public morality.

 In response, Wilde aggressively defended his novel and art in correspondence with the British press, although he personally made excisions of some of the most
controversial material when revising and lengthening the story for book publication the following year.
</item 9>

<item 10>
Dorian Gray (disambiguation)
Dorian Gray is the main character of the novel "The Picture of Dorian Gray" (1890) by Oscar Wilde.
</item 10>
=== END ITEM id=c98a2d88973b036b ===
