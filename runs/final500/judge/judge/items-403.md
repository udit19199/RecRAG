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

=== ITEM id=7049b2996ecd41a1 ===
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
Margaret Landon
Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of
the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.

 In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.

 A later work, "Never Dies the Dream", appeared in 1949.
</item 1>

<item 2>
Mary Elizabeth Bliss
Mary Elizabeth "Betty" Taylor Bliss Dandridge, born Mary Elizabeth Taylor (April 20, 1824 – July 25, 1909), was the youngest of the three surviving daughters of
President Zachary Taylor (1849-1850) and Margaret Mackall Smith Taylor.
</item 2>

<item 3>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 3>

<item 4>
At Land
At Land (1944) is a 15-minute silent experimental film written, directed by, and starring Maya Deren.

 It has a dream-like narrative in which a woman, played by Deren, is washed up on a beach and goes on a strange journey encountering other people and other
versions of herself.

 Deren once said that the film is about the struggle to maintain one's personal identity.
</item 4>

<item 5>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 5>

<item 6>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 6>

<item 7>
Elizabeth Taylor
Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.

 She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.

 She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.

 In 1999, the American Film Institute named her the seventh-greatest female screen legend.
</item 7>

<item 8>
Sam Taylor (director)
Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.

 Taylor is best known for his comedic directorial work with Harold Lloyd and Mary Pickford, and also later worked with Laurel and Hardy.
</item 8>

<item 9>
Anna and the King of Siam (novel)
Anna and the King is a 1944 semi-fictionalized biographical novel by Margaret Landon.
</item 9>

<item 10>
Krupp Diamond
The Elizabeth Taylor Diamond, formerly known as the Krupp Diamond, is a 33.19 carat diamond that was bought by Richard Burton for his wife, Elizabeth Taylor in
1968.

 The Krupp diamond was one of a number of significant pieces of jewellery owned by Taylor, her collection also included the 68 carat Taylor–Burton Diamond,
which was bought by the couple in 1969.

 The Krupp diamond was sold by Taylor's estate in 2011 for $8.8 million.
</item 10>
=== END ITEM id=7049b2996ecd41a1 ===

=== ITEM id=52b0623798068810 ===
Gold sentences and facts:
Sentence 0:  He is owner and chef of "Dear.
  Fact 0.0: He is the owner of Dear.
  Fact 0.1: He is the chef of Dear.
Sentence 1:  It airs on JTBC on Mondays at 21:30 (KST) beginning November 17, 2014.
  Fact 1.0: The show airs on JTBC.
  Fact 1.1: The show airs on Mondays at 21:30 (KST).
  Fact 1.2: The show begins airing on November 17, 2014.

Retrieved items:
<item 1>
Cook Representative
Cook Representative (), also known as National Chef Team, is a 2016 South Korean cooking-variety program starring Kim Sung-joo, Ahn Jung-hwan, Kang Ho-dong,
Choi Hyun-seok, Sam Kim, Lee Won-il and, , is the spin-off of "Please Take Care of My Refrigerator".

 It aired on JTBC during Wednesdays at 22:50 (KST) beginning February 17, 2016.

 The series aired its last episode August 10, 2016 after concluding the finals match of the World Championship.
</item 1>

<item 2>
Please Take Care of My Refrigerator
Please Take Care of My Refrigerator () is a 2014 South Korean cooking-variety program starring by many chefs and celebrity guests.

 It airs on JTBC on Mondays at 21:30 (KST) beginning November 17, 2014.
</item 2>

<item 3>
Ms Panda and Mr Hedgehog
Panda and Hedgehog () is a 2012 South Korean romantic comedy television series starring Lee Donghae of Super Junior and Yoon Seung-ah.

 Produced by Song Hae-sung's TV production venture Lion Fish, it aired on Channel A from August 18 to October 7, 2012 on Saturdays and Sundays at 21:55 for 16
episodes.
</item 3>

<item 4>
Black (TV series)
Black () is an upcoming fantasy-thriller South Korean television series starring Song Seung-heon, Go Ara, Lee El and Kim Dong-jun.

 It is set to air on OCN starting October 14, 2017 on Saturdays and Sundays at 22:20 (KST) time slot, replacing "Save Me".
</item 4>

<item 5>
Lee Won-il (born 1979) is a South Korean chef and TV personality, owner-chef of Dear. Bread, and a cast member of *Please Take Care of My Refrigerator*. He also
appeared on its 2016 spin-off *Cook Representative*. The entries describe South Korean crime and thriller films from 2007–2017, including *The Deal*, *The Boys
Who Cried Wolf*, *Deep Trap*, *A Violent Prosecutor*, *Office*, and *Fatal
</item 5>

<item 6>
Baek Jong-won's Food Truck
Baek Jong-won's Food Truck () is a South Korean cooking-variety program which has been broadcast since August 28, 2015.

 The program is currently hosted by Baek Jong-won and Kim Sung-joo, it airs on SBS every Friday at 23:20 (KST).
</item 6>

<item 7>
Lee Won-il
Lee Won-il (; born December 12, 1979) is a South Korean chef and television personality.

 He is currently a cast member in the variety show "Please Take Care of My Refrigerator".

 He is owner and chef of "Dear.

 bread".
</item 7>

<item 8>
Door to Door (Modern Family)
"Door to Door" is the fourth episode of the third season of the American sitcom "Modern Family", and the series' 52nd episode overall.

 "Door to Door" first aired on October 5, 2011, on ABC.

 The episode was written by Bill Wrubel and directed by Chris Koch.
</item 8>

<item 9>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 9>

<item 10>
The Return of Chef
"The Return of Chef" is the first episode in the tenth season of the American animated television series "South Park".

 The 140th episode of the series overall, it first aired on Comedy Central in the United States on March 22, 2006.

 The episode was the first after the departure of actor Isaac Hayes, who voiced the character Chef.

 Hayes, a Scientologist, left after a falling-out with the creators over their treatment of Scientology in the previous season's episode "Trapped in the
Closet".

 Scientology has been accused of using brainwashing techniques, and the episode portrays Chef as having been brainwashed.
</item 10>
=== END ITEM id=52b0623798068810 ===

=== ITEM id=39ba25d261fde611 ===
Gold sentences and facts:
Sentence 0: Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.
  Fact 0.0: Marcel Émile Verdet was born on 13 March 1824.
  Fact 0.1: Marcel Émile Verdet died on 3 June 1866.
  Fact 0.2: Marcel Émile Verdet was a French physicist.
Sentence 1:  Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de physique".
  Fact 1.0: Marcel Émile Verdet did much to champion the early theory of the conservation of energy in France.
  Fact 1.1: Marcel Émile Verdet championed this theory in France through his editorial supervision of the 'Annales de chimie et de physique'.
Sentence 2: Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the title "Annales de chimie".
  Fact 2.0: Annales de chimie et de physique is French for 'Annals of Chemistry and of Physics'.
  Fact 2.1: Annales de chimie et de physique is a scientific journal.
  Fact 2.2: Annales de chimie et de physique was founded in Paris, France, in 1789.
  Fact 2.3: Annales de chimie et de physique was founded under the title 'Annales de chimie'.

Retrieved items:
<item 1>
The Earrings of Madame de…
The Earrings of Madame de… (French: Madame de… ] ) is a 1953 drama film directed by Max Ophüls, adapted from Louise Leveque de Vilmorin's period novel by
Ophüls, Marcel Archard and Annette Wadement.

 The film is considered a masterpiece of the 1950s French cinema.

 Andrew Sarris called it "the most perfect film ever made".

 Ophüls said the story's construction attracted him, stating "there is always the same axis around which the action continually turns like a carousel.

 A tiny, scarcely visible axis: a pair of earrings."
</item 1>

<item 2>
Guglielmo Verdirame
Guglielmo Verdirame (born in Reggio di Calabria, Italy) is a Professor of International Law at King's College London in the Department of War Studies and the
School of Law.

 He was previously a university lecturer in law at the University of Cambridge, Faculty of Law, and a Fellow of the Lauterpacht Centre for International Law.

 He practises as a barrister at 20 Essex Street Chambers.
</item 2>

<item 3>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 3>

<item 4>
Annales de chimie et de physique
Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the
title "Annales de chimie".

 One of the early editors was the French chemist Antoine Lavoisier.

 In 1815, it became the "Annales de chimie et de physique", and was published under that name for the next 100 years.
</item 4>

<item 5>
Index of Middle English Verse
The Index of Middle English Verse (IMEV) is a bibliographic index of poetry in Middle English.

 Its first print publication, in 1943, was an extension of Carleton Brown's "Register of Middle English Religious & Didactic Verse", augmented by the inclusion
of secular verse.

 This edition, edited by Brown and Rossell Hope Robbins, contained entries for over 4000 Middle English poems in more than 2000 manuscripts.

 In 1965 the index was supplemented by Robbins and John L. Cutler.

 The IMEV is available online through Virginia Polytechnic Institute and State University.
</item 5>

<item 6>
Villard (imprint)
Villard, also known as Villard Books, is a publishing imprint of Random House, one of the largest publishing companies in the world.

 It was founded in 1983.

 Villard began as an independent imprint of Random House and is currently a sub-imprint of Ballantine Books, itself an imprint of Random House.

 It was named after a Stanford White brownstone mansion on Madison Avenue that was the home of Random House for twenty years.
</item 6>

<item 7>
Around the World in Eighty Days
Around the World in Eighty Days (French: "Le tour du monde en quatre-vingts jours" ) is a classic adventure novel by the French writer Jules Verne, published in
1873.

 In the story, Phileas Fogg of London and his newly employed French valet Passepartout attempt to circumnavigate the world in 80 days on a £20,000 wager (the
approximate equivalent of £2 million in 2016) set by his friends at the Reform Club.

 It is one of Verne's most acclaimed works .
</item 7>

<item 8>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 8>

<item 9>
Entente Cordiale (opera)
Entente Cordiale is a comic opera in one act by Ethel Smyth with an English-language libretto by Smyth, who describes the work as "a post-war comedy in one act
(founded on fact)".

 It was first performed by students at the Royal College of Music in London on 22 July 1925.
</item 9>

<item 10>
The Picture of Dorian Gray
The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published complete in the July 1890 issue of "Lippincott's Monthly Magazine".

 The magazine's editor feared the story was indecent, and without Wilde's knowledge, deleted roughly five hundred words before publication.

 Despite that censorship, "The Picture of Dorian Gray" offended the moral sensibilities of British book reviewers, some of whom said that Oscar Wilde merited
prosecution for violating the laws guarding the public morality.

 In response, Wilde aggressively defended his novel and art in correspondence with the British press, although he personally made excisions of some of the most
controversial material when revising and lengthening the story for book publication the following year.
</item 10>
=== END ITEM id=39ba25d261fde611 ===

=== ITEM id=26c4468a9240066d ===
Gold sentences and facts:
Sentence 0: Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
  Fact 0.0: Milium's vernacular name is milletgrass.
  Fact 0.1: Milium is a genus of plants in the grass family.
  Fact 0.2: Milium plants are Eurasian, North American, and North African.
Sentence 1: Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.
  Fact 1.0: Eucommia is a genus of small trees.
  Fact 1.1: Eucommia trees are now native to China.
  Fact 1.2: The fossil record of Eucommia shows a much wider distribution.
Sentence 2:  The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in herbology such as traditional Chinese medicine.
  Fact 2.0: Eucommia has a single living species, Eucommia ulmoides.
  Fact 2.1: Eucommia ulmoides is near threatened in the wild.
  Fact 2.2: Eucommia ulmoides is widely cultivated in China for its bark.
  Fact 2.3: Eucommia ulmoides is highly valued in herbology, such as traditional Chinese medicine.

Retrieved items:
<item 1>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 1>

<item 2>
Lithocarpus
Lithocarpus is a genus in the beech family Fagaceae, differing from "Quercus" in the erect spikes of insect-pollinated male flowers and the short styles with
punctate stigmas on the female flowers.

 The World Checklist (see link below) accepts 334 species, though some other texts suggest as few as 100 species.

 About 100 Asian species of the genus were formerly treated in the genus "Pasania".

 All are native to east and southeast Asia.

 These Asian species do not have a well-known English vernacular name, though the generic term stone oak has been proposed.
</item 2>

<item 3>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 3>

<item 4>
Wild boar
The wild boar ("Sus scrofa"), also known as the wild swine or Eurasian wild pig, is a suid native to much of Eurasia, North Africa, and the Greater Sunda
Islands.

 Human intervention has spread its distribution further, making the species one of the widest-ranging mammals in the world, as well as the most widely spread
suiform.

 Its wide range, high numbers, and adaptability mean that it is classed as least concern by the IUCN and it has become an invasive species in part of its
introduced range.

 The animal probably originated in Southeast Asia during the Early Pleistocene, and outcompeted other suid species as it spread throughout the Old World.
</item 4>

<item 5>
Lepidozamia comprises two Australian rainforest cycads (2n=18). Orange Field, a Sri Lankan tea factory founded in 1950, makes about 500,000 kg of black tea
annually for Middle Eastern and CIS markets, sourcing from 750 smallholders. Lithocarpus has 334 accepted East/Southeast Asian species; three Java endemics
listed here are endangered, while L. glaber occurs in Japan, China and Taiwan. Oaks (Querc
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
Eucommia
Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.

 The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in
herbology such as traditional Chinese medicine.
</item 7>

<item 8>
Ocimum
Ocimum is a genus of aromatic annual and perennial herbs and shrubs in the family Lamiaceae, native to the tropical and warm temperate regions of all 6
inhabited continents, with the greatest number of species in Africa.

 Its best known species are the cooking herb Cooking basil, "O. basilicum" and the medicinal herb Tulsi (holy basil), "O. tenuiflorum".
</item 8>

<item 9>
Milium (plant)
Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
</item 9>

<item 10>
Eucryphia: seven mostly evergreen South American/Australian trees; classified in Cunoniaceae. Bothriochloa ischaemum: widespread grass; var. songarica invasive
in Texas. Eriogonum buckwheats: genus diverse across North America; E. heracleoides inhabits western U.S.; E. diclinum forms low mats on serpentine soils; E.
crosbyae occurs in Oregon/Nevada.
</item 10>
=== END ITEM id=26c4468a9240066d ===

=== ITEM id=067672da73c723e4 ===
Gold sentences and facts:
Sentence 0: And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
  Fact 0.0: And There's More is a comedy sketch show.
  Fact 0.1: And There's More stars Jimmy Cricket.
  Fact 0.2: And There's More was produced by Central Independent Television.
  Fact 0.3: And There's More was produced for ITV.
  Fact 0.4: And There's More was produced from 1985 till 1988.
Sentence 1:  He currently lives with his family in Rochdale, North West England.
  Fact 1.0: The person described in this passage currently lives with his family in Rochdale, North West England.
  Fact 1.1: The person described in this passage currently lives in Rochdale.
  Fact 1.2: Rochdale is in North West England.

Retrieved items:
<item 1>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 1>

<item 2>
And There's More
And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
</item 2>

<item 3>
Hotel Kenmore Hall
Hotel Kenmore Hall is a 22-story single room occupancy hotel located at 145 East 23rd Street in the Gramercy section of Manhattan, designed by architect Maurice
Deutsch and constructed in 1927.

 Author Nathanael West lived and worked at the hotel as a night manager in the early years after the hotel opened; one of West's real-life experiences at the
hotel inspired the incident between Romola Martin and Homer Simpson that would later appear in "The Day of the Locust" (1939).

 West allowed friends like Edmund Wilson, Erskine Caldwell, S. J. Perelman and Maxwell Bodenheim free room and meals.

 Dashiell Hammett finished "The Maltese Falcon" here
</item 3>

<item 4>
The entries describe U.S. communities, infrastructure and historic sites: Clermont, Iowa, has Montauk and “The Straight Story” filming locations; Clearfield
County’s rural Pennsylvania school district serves about 19,000. North Dakota’s H-T Ranch was a major cattle and horse operation. Pennsylvania’s Toland began as
a clay-company village; New Jersey entries cover suburban developments and mill communities. Other highlights include North Carolina’s Lucas Mansion, D.C.’s
Congress Heights neighborhoods and Metro station
</item 4>

<item 5>
Standard Star Building
The Standard Star Building is a historic commercial structure located in the Downtown section of New Rochelle, Westchester County, New York.

 The building, designed by architect Lawrence J. Barnard, was completed in 1924 and is an architecturally significant example of the Italian Renaissance style
in New Rochelle.

 Although it has been altered, these changes occurred only on the Le Count Place façade and the interior of the building.

 The original Standard Star building exterior remains largely unchanged.

 It is further historically significant for its association with a long-published New Rochelle newspaper covering life in New Rochelle.
</item 5>

<item 6>
John Garabedian
John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).

 He is best known as the creator and former long-time host of "Open House Party".

 He has been involved in Massachusetts radio and television stations for over fifty years.

 Garabedian currently owns three homes, all in New England: one in Southborough, Massachusetts, another on Cape Cod, and a cabin-like home in Cabot, Vermont.
</item 6>

<item 7>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 7>

<item 8>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 8>

<item 9>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 9>

<item 10>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 10>
=== END ITEM id=067672da73c723e4 ===

=== ITEM id=3a77e41de8d3375a ===
Gold sentences and facts:
Sentence 0: The Apple Dumpling Gang is a 1975 American comedy-western film produced by Walt Disney Productions about a slick gambler named Russell Donovan (Bill Bixby) who is duped into taking care of a group of orphans who eventually strike gold during the California Gold Rush.
  Fact 0.0: The Apple Dumpling Gang is a 1975 American comedy-western film.
  Fact 0.1: The Apple Dumpling Gang was produced by Walt Disney Productions.
  Fact 0.2: The Apple Dumpling Gang is about a slick gambler named Russell Donovan.
  Fact 0.3: Russell Donovan is played by Bill Bixby.
  Fact 0.4: Russell Donovan is duped into taking care of a group of orphans.
  Fact 0.5: The group of orphans eventually strikes gold during the California Gold Rush.
Sentence 1: Heavyweights is a 1995 American comedy-drama film directed by Steven Brill and written by Brill with Judd Apatow.
  Fact 1.0: Heavyweights is a 1995 American comedy-drama film.
  Fact 1.1: Heavyweights was directed by Steven Brill.
  Fact 1.2: Heavyweights was written by Steven Brill with Judd Apatow.

Retrieved items:
<item 1>
The Apple Dumpling Gang (film)
The Apple Dumpling Gang is a 1975 American comedy-western film produced by Walt Disney Productions about a slick gambler named Russell Donovan (Bill Bixby) who
is duped into taking care of a group of orphans who eventually strike gold during the California Gold Rush.
</item 1>

<item 2>
The Apple Dumpling Gang
The Apple Dumpling Gang is a 1971 novel by Jack Bickham, about a group of orphaned children during the California gold rush.

 They encounter a gambler who reluctantly helps them, as well as a pair of hapless robbers who are after the gold the children have found.
</item 2>

<item 3>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 3>

<item 4>
Heavyweights
Heavyweights is a 1995 American comedy-drama film directed by Steven Brill and written by Brill with Judd Apatow.

 The film centers around a fat camp for kids that is taken over by a fitness guru named Tony Perkis (Ben Stiller).
</item 4>

<item 5>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 5>

<item 6>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 6>

<item 7>
Heavy Traffic
Heavy Traffic is a 1973 American adult animated comedy-drama film written and directed by Ralph Bakshi.

 The film, which begins, ends, and occasionally combines with live-action, explores the often surreal fantasies of a young New York cartoonist named Michael
Corleone, using pinball imagery as a metaphor for inner-city life.

 "Heavy Traffic" was Bakshi and producer Steve Krantz's follow-up to the film "Fritz the Cat".

 Though producer Krantz made varied attempts to produce an R-rated film, "Heavy Traffic" was given an X rating by the MPAA.

 The film received positive reviews and is widely considered to be Bakshi's biggest critical success.
</item 7>

<item 8>
The Sasquatch Gang
The Sasquatch Gang (also known as The Sasquatch Dumpling Gang) is a 2006 comedy film written and directed by Tim Skousen, the first assistant director on
"Napoleon Dynamite".
</item 8>

<item 9>
The Apple Dumpling Gang Rides Again
The Apple Dumpling Gang Rides Again is a 1979 American comedy-western film produced by Walt Disney Productions and a sequel to "The Apple Dumpling Gang" (1975),
starring the comedy duo of Tim Conway and Don Knotts reprising their respective roles as Amos and Theodore.

 The film also stars Tim Matheson, Harry Morgan, and Kenneth Mars.

 Ruth Buzzi appears in a small cameo as a wild farsighted woman.

 Robert Totten, who directed installments of "Gunsmoke", also had a small part in the film.
</item 9>

<item 10>
Underdog (film)
Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.

 Directed by Frederik Du Chau and written by Joe Piscatella, Adam Rifkin, and Craig A. Williams, the film stars Jim Belushi, Peter Dinklage, John Slattery, and
Patrick Warburton with the voice talents of Jason Lee, Amy Adams, and Brad Garrett.

 Unlike the TV series, the Underdog character is portrayed as a regular dog rather than an anthropomorphic one.

 Underdog, voiced by Jason Lee, was played by a lemon beagle named Leo sporting a red sweater and a blue cape.

 The film grossed $65.3 million worldwide.
</item 10>
=== END ITEM id=3a77e41de8d3375a ===

=== ITEM id=dbbb8fbc56b2044d ===
Gold sentences and facts:
Sentence 0: The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between 1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
  Fact 0.0: The New York – New Jersey Line War is also known as the N.J. Line War.
  Fact 0.1: The New York – New Jersey Line War refers to a series of skirmishes and raids.
  Fact 0.2: The skirmishes and raids took place for over half a century.
  Fact 0.3: The skirmishes and raids took place between 1701 and 1765.
  Fact 0.4: The skirmishes and raids took place at the disputed border between two American colonies.
  Fact 0.5: The two American colonies were the Province of New York and the Province of New Jersey.
Sentence 1: The Province of New York (1664–1776) was a British proprietary colony and later royal colony on the northeast coast of North America.
  Fact 1.0: The Province of New York existed from 1664 to 1776.
  Fact 1.1: The Province of New York was a British proprietary colony.
  Fact 1.2: The Province of New York later became a royal colony.
  Fact 1.3: The Province of New York was on the northeast coast of North America.

Retrieved items:
<item 1>
Stamp Act 1765
The Stamp Act of 1765 (short title "Duties in American Colonies Act 1765"; 5 George III, c. 12) was an Act of the Parliament of Great Britain that imposed a
direct tax on the colonies of British America and required that many printed materials in the colonies be produced on stamped paper produced in London, carrying
an embossed revenue stamp.

 Printed materials included legal documents, magazines, playing cards, newspapers, and many other types of paper used throughout the colonies.

 Like previous taxes, the stamp tax had to be paid in valid British currency, not in colonial paper money.

 The purpose of the tax was to help pay for troops stationed in North America after the British victory in the Seven Years' War and its North American theater
of the French and Indian War.

 The Americans said that there was no military need for the soldiers because there were no foreign enemies on the continent, and the Americans had always
protected themselves against Indians.

 They suggested that it was actually a matter of British patronage to surplus British officers and career soldiers who should be paid by London.
</item 1>

<item 2>
In 1780, Indigenous allies ambushed Jacobus Westervelt’s 41-person Dutch settler caravan in Kentucky; 17 died, including 10 Westervelts. New Jersey evolved from
Dutch and English rule, split into East and West Jersey, reunited in 1702, and adopted a constitution in 1776. Pennsylvania and New York were proprietary
colonies; charter colonies included Rhode Island, Connecticut, and Massachusetts Bay. Other topics include Great Dismal Swamp maroons, Valley Forge
</item 2>

<item 3>
Province of New York
The Province of New York (1664–1776) was a British proprietary colony and later royal colony on the northeast coast of North America.

 As one of the Thirteen Colonies, New York achieved independence and worked with the others to found the United States.
</item 3>

<item 4>
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
</item 4>

<item 5>
New York – New Jersey Line War
The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between
1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
</item 5>

<item 6>
Canada under British rule
Canada first came under British rule with the Treaty of Paris (1763) which ceded New France, of which Canada was a part, to the British Empire.

 Gradually, other territories, colonies, and provinces that were part of British North America would be added to Canada.

 The Royal Proclamation of 1763 enlarged the colony of Canada under the name of the Province of Quebec, which with the Constitutional Act 1791 became known as
The Canadas.

 With the Act of Union 1840 Upper and Lower Canada were joined to become the United Province of Canada.

 Later, with Confederation in 1867, the British maritime colonies of New Brunswick and Nova Scotia were joined with the British colony of Canada to form the
Dominion of Canada, which was subsequently divided into four provinces, Ontario, Quebec, New Brunswick, and Nova Scotia.

 A number of other British colonies, such as Newfoundland and British Columbia, and large territories such as Rupert's Land initially remained outside of the
newly formed federation.

 Over time, the remaining colonies and territories within British North America came under the control of Canada until the current geographic extent of the
country was reached when Newfoundland and Labrador joined Canada in 1949.
</item 6>

<item 7>
Charter colony
Charter colony is one of three classes of colonial government established in the 17th century English colonies in North America, the other classes being
proprietary colony and royal colony.

 The colonies of Rhode Island, Connecticut, and Massachusetts Bay were charter colonies.

 In a charter colony, Britain granted a charter to the colonial government establishing the rules under which the colony was to be governed.

 The charters of Rhode Island and Connecticut granted the colonists significantly more political liberty than other colonies.

 Rhode Island and Connecticut continued to use their colonial charters as their State constitutions after the American Revolution.
</item 7>

<item 8>
Province of New Jersey
The Province of New Jersey was one of the Middle Colonies of Colonial America and became the U.S. state of New Jersey in 1776.

 The province had originally been settled by Europeans as part of New Netherland, but came under English rule after the surrender of Fort Amsterdam in 1664,
becoming a proprietary colony.

 The English then renamed the province after the Isle of Jersey in the English Channel.

 The Dutch Republic reasserted control for a brief period in 1673–1674.

 After that it consisted of two political divisions, East Jersey and West Jersey, until they were united as a royal colony in 1702.

 The original boundaries of the province were slightly larger than the current state, extending into a part of the present state of New York, until the border
was finalized in 1773.
</item 8>

<item 9>
Province of Pennsylvania
The Province of Pennsylvania, also known as the Pennsylvania Colony, was founded in English North America by William Penn on March 4, 1681 as dictated in a
royal charter granted by King Charles II.

 The name Pennsylvania, which translates roughly as "Penn's Woods", was created by combining the Penn surname (in honor of William's father, Admiral Sir William
Penn) with the Latin word "sylvania", meaning "forest land."

 The Province of Pennsylvania was one of the two major Restoration colonies, the other being the Province of Carolina.

 The proprietary colony's charter remained in the hands of the Penn family until the American Revolution, when the Commonwealth of Pennsylvania was created and
became one of the original thirteen states.
</item 9>

<item 10>
Proprietary colony
A proprietary colony was a type of British colony mostly in North America and the Caribbean in the 17th century.

 In the British Empire, all land belonged to the ruler, and it was his prerogative to divide.

 Therefore, all colonial properties were partitioned by royal charter into one of four types: proprietary, royal, joint stock, or covenant.

 King Charles II used the proprietary solution to reward allies and focus his own attention on Britain itself.

 He offered his friends colonial charters which facilitated private investment and colonial self-government.

 The charters made the proprietor the effective ruler, albeit one ultimately responsible to English law and the king.

 Charles II gave New Netherland to his younger brother The Duke of York, who named it New York.

 He gave an area to William Penn who named it Pennsylvania.
</item 10>
=== END ITEM id=dbbb8fbc56b2044d ===

=== ITEM id=bfc2594f6abb2ccb ===
Gold sentences and facts:
Sentence 0: Marc Predka (born April 21, 1975) better known by his stage name Tha Trademarc, is an American hip hop artist.
  Fact 0.0: Marc Predka was born April 21, 1975.
  Fact 0.1: Marc Predka is better known by his stage name Tha Trademarc.
  Fact 0.2: Marc Predka is an American hip hop artist.
Sentence 1:  He first came to prominence when he and his first cousin, WWE wrestler John Cena, collaborated on the 2005 album "You Can't See Me", including Cena's entrance theme "The Time Is Now".
  Fact 1.0: Marc Predka first came to prominence when he collaborated with his first cousin, John Cena.
  Fact 1.1: John Cena is a WWE wrestler.
  Fact 1.2: Marc Predka and John Cena collaborated on the 2005 album 'You Can't See Me'.
  Fact 1.3: The album 'You Can't See Me' includes Cena's entrance theme 'The Time Is Now'.
Sentence 2: "The Time Is Now" is a song by professional wrestler, actor, and hip hop musician John Cena.
  Fact 2.0: 'The Time Is Now' is a song by John Cena.
  Fact 2.1: John Cena is a professional wrestler.
  Fact 2.2: John Cena is an actor.
  Fact 2.3: John Cena is a hip hop musician.

Retrieved items:
<item 1>
Tha Trademarc
Marc Predka (born April 21, 1975) better known by his stage name Tha Trademarc, is an American hip hop artist.

 He first came to prominence when he and his first cousin, WWE wrestler John Cena, collaborated on the 2005 album "You Can't See Me", including Cena's entrance
theme "The Time Is Now".

 He also appeared in the music videos of the songs, "Bad Bad Man" and "Right Now" with Cena.

 Prior to the album, Trademarc appeared on Cena's first WWE DVD "Word Life".

 He now works as a classroom specialist for the Peace Corps of America.
</item 1>

<item 2>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 2>

<item 3>
The Preytells
The Preytells were an Australian indie rock band from Perth, formed in 2004 by Jessica Bennett (a.k.a. Audrey Tell) on lead guitar and backing vocals, Simon
Okely (a.k.a. Will Tell) on guitar and lead vocals, Jaclyn Pearson on drums and percussion and Cameron Stewart on bass guitar and backing vocals.

 Their sole album, "Flood Songs/June Songs", was issued in September 2009; before they disbanded later that year.

 Their single, "Shout!"

 ( May 2008), was nominated for WAM Song of the Year in the Pop category in 2008; while, "Lord Hold My Hand", was nominated for the same category in 2009.
</item 3>

<item 4>
Matt Thiessen
Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist,
pianist, and primary songwriter for the Christian rock band Relient K.

 With Relient K, he has released eight full-length albums, including three that were certified Gold, and three that peaked in the top twenty on the "Billboard"
200.

 Outside of his work with Relient K, Thiessen maintains a low-profile side project called Matthew Thiessen and the Earthquakes.

 In 2009, he co-produced and collaborated on Owl City's album "Ocean Eyes".
</item 4>

<item 5>
Danja (record producer)
Floyd Nathaniel "Nate" Hills (born February 22, 1982), professionally known as Danja , is an American record producer and songwriter from Virginia Beach,
Virginia.

 He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah
Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink,
T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
</item 5>

<item 6>
Den Za Nas
Den Za Nas (in Macedonian: Ден за нас ) is the title of the fourth Macedonian language album by the Macedonian superstar Toše Proeski released in 2004.

 It is a 2-CD set which contains all of the songs performed on the National final in order to select a song for the Eurovision Song Contest 2004 with which
Proeski would represent Macedonia in the contest.

 The winning song was "Life" which qualified for the final and finished on the 14th place in the final.

 Some extra songs such as "Dzvezdo Severnice", "Me Sakaš So Zborovi", "Ima Li Den Za Nas" and others were recorded especially for the album.
</item 6>

<item 7>
Phil Driscoll
Phil Driscoll (born November 9, 1947) is a trumpeter, singer, composer, and producer.

 He performs in varying music genres and styles which include rock and roll, rhythm and blues, and patriotic music, and is best known for his work in Christian
music and his longterm Christian ministry.

 In 1985, Driscoll won the Grammy Award for Best Gospel Performance – Duo/Group for a duet with Debby Boone on "Keep the Flame Burning", and he has been
nominated for three additional Grammys, two for Best Gospel Performance – Male and one for Best Gospel/Pop Album.

 He has also won three Dove Awards for his music, and the 1999 Christian Country Music Association Award for Best Musician.

 In 2006, Driscoll was found guilty on 2 counts of tax evasion and one count of conspiracy, and was sentenced to serve one year in Federal prison, beginning on
March 14, 2007.
</item 7>

<item 8>
Zach Crowell
Zach Crowell is a Grammy Award-winning songwriter and record producer, who was born and raised in Nashville, Tennessee.

 Crowell is involved in a joint publishing deal with Warner/Chappell Music, Combustion Music and Taperoom Music.
</item 8>

<item 9>
Mike Mattison
Mike Mattison is an American musician and vocalist of the Grammy Award-winning blues rock/soul group, the Tedeschi Trucks Band, as well as lead vocalist and
co-founder of the blues rock trio Scrapomatic.

 Mattison's vocal sound has been described as "strong," with an "expressive blues voice".

 As lead vocalist of Scrapomatic, he picked up a nomination for Minnesota Music Awards best male vocalist, and both he and co-founder Paul Olsen were also
nominated for best R&B Group.

 Mattison was previously the lead vocalist of the Grammy Award winning Derek Trucks Band and has been a main songwriter of all three bands.

 He is also an active essayist who publishes on music and poetry.

 Since 2013 Mattison and Ernest Suarez have edited “Hot Rocks: Songs and Verse,” an ongoing feature in Five Points: A Journal of Literature and Art.
</item 9>

<item 10>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 10>
=== END ITEM id=bfc2594f6abb2ccb ===

=== ITEM id=189d9bd04d97ffed ===
Gold sentences and facts:
Sentence 0: Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.
  Fact 0.0: Jean-François André Clervoy was born on 19 November 1958.
  Fact 0.1: Jean-François André Clervoy is a French engineer.
  Fact 0.2: Jean-François André Clervoy is a CNES astronaut.
  Fact 0.3: Jean-François André Clervoy is an ESA astronaut.
Sentence 1: Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
  Fact 1.0: Prof. Dr. Ulrich Hans Walter was born on February 9, 1954.
  Fact 1.1: Prof. Dr. Ulrich Hans Walter is a German physicist.
  Fact 1.2: Prof. Dr. Ulrich Hans Walter is a German engineer.
  Fact 1.3: Prof. Dr. Ulrich Hans Walter is a former DFVLR astronaut.

Retrieved items:
<item 1>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 1>

<item 2>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 2>

<item 3>
Walter Cunningham
Ronnie Walter Cunningham (born March 16, 1932), (Col, USMCR, Ret.)

, better known as Walter Cunningham, is a retired American astronaut.

 In 1968, he was a Lunar Module Pilot on the Apollo 7 mission.

 He was NASA's third civilian astronaut (after Neil Armstrong and Elliot See), and has also been a fighter pilot, physicist, entrepreneur, venture capitalist,
author of "The All-American Boys", lecturer, and host of the radio show "Lift-off to Logic".

 Walter Cunningham attended UCLA.
</item 3>

<item 4>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 4>

<item 5>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 5>

<item 6>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 6>

<item 7>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 7>

<item 8>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 8>

<item 9>
Philippe Perrin
Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
</item 9>

<item 10>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 10>
=== END ITEM id=189d9bd04d97ffed ===

=== ITEM id=0ffe7e2a7c0e28e2 ===
Gold sentences and facts:
Sentence 0:  The film concentrated on Ellsworth "Bumpy" Johnson (Laurence Fishburne), Dutch Schultz (Tim Roth), and Lucky Luciano (Andy García).
  Fact 0.0: The film concentrated on Ellsworth "Bumpy" Johnson.
  Fact 0.1: Laurence Fishburne played Ellsworth "Bumpy" Johnson in the film.
  Fact 0.2: The film concentrated on Dutch Schultz.
  Fact 0.3: Tim Roth played Dutch Schultz in the film.
  Fact 0.4: The film concentrated on Lucky Luciano.
  Fact 0.5: Andy García played Lucky Luciano in the film.
Sentence 1:  Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.
  Fact 1.0: Lucky Luciano is considered the father of modern organized crime in the United States.
  Fact 1.1: Lucky Luciano is considered the father of modern organized crime in the United States for establishing the first Commission.
  Fact 1.2: Lucky Luciano established the first Commission.

Retrieved items:
<item 1>
Lucky Luciano
Charles "Lucky" Luciano ( ; born Salvatore Lucania November 24, 1897 – January 26, 1962) was an Italian-American mobster and crime boss.

 Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.

 He was also the first official boss of the modern Genovese crime family.

 He was, along with his associates, instrumental in the development of the National Crime Syndicate.
</item 1>

<item 2>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 2>

<item 3>
Illtown
Dante and his girlfriend Micky run a very profitable drug operation in a seaside town, aided and abetted by a host of teens who sell the smack at discos around
town, as well as by Lucas, a corrupt cop who's on the take.

 Their downfall comes when they suspect one of the boys, Pep, of ripping them off, and his accidental death causes disloyalty among the teens, who suspect Dante
offed them.

 All of this is perfect for the return of Gabriel, a one-time partner of Dante, who has just been released from jail, and has an almost angelic demeanor and the
certainty that he can fix everyone's lives.
</item 3>

<item 4>
Ang Utol Kong Hoodlum
Ang Utol Kong Hoodlum (lit.

 "My Brother is a Hoodlum") is a Filipino drama series developed for TV5 created by Deo J. Fajardo.

 It stars JC de Vera and Jasmine Curtis-Smith.

 It is a remake of the original movie where Robin Padilla played the role of Ben, a hoodlum, and Vina Morales, as Bing.

 The movie was first released in 1991, then a sequel entitled "Miss na Miss Kita: Ang Utol Kong Hoodlum 2" was made the following year.

 It is produced by Vic Del Rosario Jr., and Manuel V. Pangilinan and it also marks as the first primetime series produced by Viva Television for TV5 after a
decade.
</item 4>

<item 5>
James Marcello
James J. Marcello (born December 13, 1943), also known variously as Little Jimmy, Jimmy Light and as Jimmy the Man Marcello, is an imprisoned crime boss who was
a front boss for the Chicago Outfit criminal organization in the 1980s, 1990s and early 2000s.

 Organized crime observers identified Marcello as a figurehead during that period while the organization's day-to-day operations actually were run by John “No
Nose” DiFronzo, Joseph “Joey the Clown” Lombardo, Joseph “Joe the Builder” Andriacchi and Angelo J. LaPietra.

 Marcello is the Cousin of Michael Hayes Marcello, and a cousin of Robert Urbinatti.

 He is also the father of Rocco Marcello.
</item 5>

<item 6>
The Yakuza
The Yakuza is a 1974 Japanese-American neo-noir gangster film directed by Sydney Pollack, written by Leonard Schrader, Paul Schrader, and Robert Towne.

 The film is about a man (Robert Mitchum) who returns to Japan after several years away in order to rescue his friend's kidnapped daughter.

 Following a lackluster initial release, the film has since gained a cult following.
</item 6>

<item 7>
Hoodlum (film)
Hoodlum is a 1997 American crime drama film that gives a fictionalized account of the gang war between the Italian/Jewish mafia alliance and the Black gangsters
of Harlem that took place in the late 1920s and early 1930s.

 The film concentrated on Ellsworth "Bumpy" Johnson (Laurence Fishburne), Dutch Schultz (Tim Roth), and Lucky Luciano (Andy García).
</item 7>

<item 8>
Joseph Todaro Jr.
Joseph "Big Joe" Todaro Jr. (born 1945 or 1946) is a Buffalo, New York businessman and former organized crime figure involved in labor racketeering,
loansharking, illegal gambling, narcotics, and murder for hire.

 Joe Todaro Jr. became a business agent for the Laborers' International Union of North America (LIUNA) Local 210.

 In 1984, Joe Todaro Jr. allegedly became the underboss of the Buffalo crime family after his father became the new head of the crime family, upon the
retirement of his predecessor Samuel "Sam the Farmer" Frangiamore.

 In 1990, Joe Todaro Jr. resigned as business agent following investigations on the local's alleged ties to organized crime.

 Outside of organized crime, Todaro operates La Nova Pizzeria, a popular pizza restaurant in Buffalo.
</item 8>

<item 9>
Hoodlum (soundtrack)
Hoodlum is the soundtrack to the 1997 American crime drama film "Hoodlum".

 It was released on August 12, 1997 by Interscope Records and consisted of a blend of hip hop and R&B music.

 The soundtrack peaked at 94 on the "Billboard" 200 and 23 on the Top R&B Albums and contained Mobb Deep's single "Hoodlum" which went to 29 on the Hot Rap
Singles.
</item 9>

<item 10>
Frank Colacurcio
Francis Colacurcio, Sr. (June 18, 1917 – July 2, 2010) was an American businessman and boss of the Seattle crime family known for running strip clubs in
Seattle, Washington.

 He gained notoriety as a subject of ongoing federal investigations into organized crime in the city and was suspected of being an organized crime boss.
</item 10>
=== END ITEM id=0ffe7e2a7c0e28e2 ===

=== ITEM id=707480a5af76d854 ===
Gold sentences and facts:
Sentence 0:  She is married to Scott Stapp, former frontman of the band Creed, and current frontman for the band Art of Anarchy.
  Fact 0.0: The woman described is married to Scott Stapp.
  Fact 0.1: Scott Stapp is a former frontman of the band Creed.
  Fact 0.2: Scott Stapp is the current frontman for the band Art of Anarchy.
Sentence 1:  The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between members.
  Fact 1.0: The band's third album, “Weathered”, was released in 2001.
  Fact 1.1: Tremonti handled bass when “Weathered” was released.
  Fact 1.2: The band disbanded in 2004.
  Fact 1.3: The band disbanded in 2004 due to increasing tension between members.

Retrieved items:
<item 1>
Jaclyn Stapp
Jaclyn Nesheiwat Stapp (born July 29, 1980) is a beauty queen, philanthropist and fashion model with pageant roots in Florida and New York.

 She is married to Scott Stapp, former frontman of the band Creed, and current frontman for the band Art of Anarchy.

 Her most notable titles include Mrs. Florida America 2008 and Miss New York USA 2004.

 She is executive director of The Scott Stapp With Arms Wide Open Foundation.
</item 1>

<item 2>
Zachary Throne
As a singer, Zachary has sung on many TV and radio jingles.

 He provided the singing voice for the character, "Mark Winkle" on the television series, "California Dreams" as well as the singing voice for the character,
"Greg Brady" in the films, "The Brady Bunch Movie" and "A Very Brady Sequel".

 Currently, he resides in Las Vegas, Nevada where he performs in many shows.

 From 2012-2014, he was the lead singer/lead guitar player/bass player for the Sin City Sinners, a group that also featured former Faster Pussycat guitarist and
co-founder Brent Muscat as well as Slash bass player, Todd Kerns.

 .

 With the group, he recorded two albums, "DIVEBAR Days Revisited" and "A Sinners Christmas 2", both released in 2013.
</item 2>

<item 3>
The Replacements (band)
The Replacements were an American rock band formed in Minneapolis, Minnesota, in 1979.

 Initially a punk rock band, they are considered pioneers of alternative rock.

 The band was composed of the guitarist and vocalist Paul Westerberg, guitarist Bob Stinson, bass guitarist Tommy Stinson and drummer Chris Mars for most of its
career.

 Following several acclaimed albums, including "Let It Be" and "Tim", Bob Stinson left the band in 1986, and Slim Dunlap joined as lead guitarist.

 Steve Foley replaced Mars in 1990.

 Towards the end of the band's career, Westerberg exerted more control over the creative output.

 The group disbanded in 1991, with the members eventually pursuing various projects.

 A reunion was announced on October 3, 2012.

 The Replacements never had significant commercial success, except for "I'll Be You", which hit number 1 on the "Billboard" Alt Rock chart, but they influenced
numerous alternative rock acts.
</item 3>

<item 4>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 4>

<item 5>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 5>

<item 6>
The Studdogs
The band was signed by Orange Recordings in Los Angeles and toured all over the country, sharing bills with the Demolition Doll Rods, Bob Log III, The
Fleshtones, and Immortal Lee County Killers and a one-time show with the Suicide Girls Burlesque Tour in Orlando.

 While in the Studdogs, Rich Evans began promoting concerts under the tag "Mutiny Productions".

 He now heads the independent "Florida's Dying" label, promoting and releasing material from the Florida-underground music scene.

 Kyle Justin joined the band on drums in early 2002 to replace Jason, who was moving to Texas.

 Kyle vacated his position as drummer in 2003 to pursue a music career in Los Angeles.

 Eric Gebhardt left the band in 2005 to pursue a solo career under the alias Red Mouth.
</item 6>

<item 7>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 7>

<item 8>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 8>

<item 9>
Creed (band)
Creed is an American rock band formed in 1993 in Tallahassee, Florida.

 The band's best-known line-up consists of lead vocalist Scott Stapp, guitarist and vocalist Mark Tremonti, bassist Brian Marshall, and drummer Scott Phillips.

 Creed released two studio albums, "My Own Prison" in 1997 and "Human Clay" in 1999, before Marshall left the band in 2000.

 The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between
members.

 Tremonti, Marshall, and Phillips went on to found Alter Bridge while Stapp followed a solo career.
</item 9>

<item 10>
Rock Star (TV series)
Rock Star is a television series produced by Mark Burnett, David Goffin, and Lisa Hennessy in which aspiring singers from all around the world competed to
become the lead singer of a featured group.

 It debuted on CBS on July 11, 2005, to mediocre ratings.

 The show was hosted by television personality and commercial spokeswoman Brooke Burke and Jane's Addiction & Red Hot Chili Peppers lead guitarist Dave Navarro.

 In Australian band INXS chose J.D. Fortune as their new lead singer.

 For , the band Rock Star Supernova chose Lukas Rossi as the lead singer of their new supergroup.
</item 10>
=== END ITEM id=707480a5af76d854 ===
