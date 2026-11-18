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

=== ITEM id=a60379d94f3b5986 ===
Gold sentences and facts:
Sentence 0:  It was created by Jay Lender and Micah Wright, former writers for "The Angry Beavers", "SpongeBob SquarePants", and "Phineas and Ferb".
  Fact 0.0: The series was created by Jay Lender.
  Fact 0.1: The series was created by Micah Wright.
  Fact 0.2: Jay Lender was a former writer for The Angry Beavers.
  Fact 0.3: Jay Lender was a former writer for SpongeBob SquarePants.
  Fact 0.4: Jay Lender was a former writer for Phineas and Ferb.
  Fact 0.5: Micah Wright was a former writer for The Angry Beavers.
  Fact 0.6: Micah Wright was a former writer for SpongeBob SquarePants.
  Fact 0.7: Micah Wright was a former writer for Phineas and Ferb.
Sentence 1:  The series chronicles the adventures and endeavors of the title character and his various friends in the fictional underwater city of Bikini Bottom.
  Fact 1.0: The series chronicles the adventures and endeavors of the title character and his various friends.
  Fact 1.1: The series is set in the fictional underwater city of Bikini Bottom.

Retrieved items:
<item 1>
SpongeBob SquarePants: Battle for Bikini Bottom
SpongeBob SquarePants: Battle for Bikini Bottom is a 2003 video game based on the animated series of the same name, developed by Heavy Iron Studios, AWE Games,
and Vicarious Visions, and published by THQ.

 The game was released for the PlayStation 2, Game Boy Advance (GBA), Xbox, and Nintendo GameCube consoles as well as a port for Microsoft Windows.

 The Microsoft Windows version of the game is a point-and-click adventure game, whereas the 6th generation platforms received a 3D platformer.

 The Game Boy Advance version received a 2D platformer.

 All versions of the game feature an original storyline, in which the player attempts to defend Bikini Bottom from an invasion of robots created by Plankton
with a machine called the Duplicatotron 3000, playing as SpongeBob in all versions, as well as Patrick and Sandy in the console versions.

 The game was released on October 31, 2003, in North America and in Europe on November 28, 2003.
</item 1>

<item 2>
Robocalypse
Robocalypse is a real-time strategy video game developed and published by Vogster Entertainment for the Nintendo DS handheld video game console.

 It has only been released in North America.

 It was created by Jay Lender and Micah Wright, former writers for "The Angry Beavers", "SpongeBob SquarePants", and "Phineas and Ferb".
</item 2>

<item 3>
SpongeBob, You're Fired
"SpongeBob, You're Fired" is a television special of the American animated television series "SpongeBob SquarePants".

 It serves as the 11th episode of the ninth season and the 189th overall episode.

 The animation directors were the supervising director, Alan Smart and Tom Yasumi, and was written by Marc Ceccarelli, Luke Brookshier, and Mr. Lawrence.

 Ceccarelli and Brookshier also served as storyboard directors.

 The episode originally aired in Greece on July 3, 2013.

 In the United States, it aired on Nickelodeon on November 11.

 In this episode of the series, which follows the adventures and endeavors of the title character and his various friends in the underwater city of Bikini
Bottom, SpongeBob gets fired from the Krusty Krab after Mr. Krabs discovers he can save a nickel by eliminating him.

 SpongeBob subsequently decides to apply at other restaurants.
</item 3>

<item 4>
SpongeBob SquarePants (season 3)
The third season of the American animated television series "SpongeBob SquarePants", created by Stephen Hillenburg, aired on Nickelodeon from October 5, 2001 to
October 11, 2004, and consists of 20 episodes.

 The series chronicles the exploits and adventures of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The season was executive produced by series creator Hillenburg, who also acted as the showrunner.

 Hillenburg halted production on the show to work on the 2004 film adaptation of the series, "The SpongeBob SquarePants Movie".

 After production on the film, Hillenburg resigned from the show as its showrunner, and appointed staff writer Paul Tibbitt to overtake the position.

 Season 3 was originally set to end the series after the release of the film, but the success prevented the series from ending, leading to a fourth season.
</item 4>

<item 5>
Rock Bottom (SpongeBob SquarePants)
"Rock Bottom" is the 34th episode of the first season of the American animated television series "SpongeBob SquarePants".

 It originally was produced in 1999 and aired on Nickelodeon in the United States on March 15, 2000.

 The series follows the adventures of the title character in the underwater city of Bikini Bottom.

 In the episode, SpongeBob becomes stranded in a abyssopelagic zone called Rock Bottom.
</item 5>

<item 6>
Band Geeks
"Band Geeks" is the second part of the 15th episode of the second season, and the second half of the 35th episode overall, of the American animated television
program "SpongeBob SquarePants".

 It originally aired on Nickelodeon in the United States on September 7, 2001.

 It was written by C. H. Greenblatt, Aaron Springer, and Merriwether Williams, and the animation was directed by Frank Weiss.

 Springer served as storyboard director, and Greenblatt served as storyboard artist.

 The song "Sweet Victory" by David Glen Eisley was featured in the episode and was later released on the album "" in 2005.
</item 6>

<item 7>
List of SpongeBob SquarePants episodes
"SpongeBob SquarePants" is an American animated television series created by marine biologist and animator Stephen Hillenburg for Nickelodeon.

 The series is set in the fictional underwater city of Bikini Bottom, and centers on the adventures and endeavors of SpongeBob SquarePants, an over-optimistic
sea sponge that annoys other characters.

 Many of the ideas for the show originated in an unpublished, educational comic book titled "The Intertidal Zone", which Hillenburg created in the mid-1980s.

 He began developing "SpongeBob SquarePants" into a television series in 1996 after the cancellation of "Rocko's Modern Life", another Nickelodeon television
series which Hillenburg previously directed.
</item 7>

<item 8>
SpongeBob SquarePants (season 4)
The fourth season of the American animated television series "SpongeBob SquarePants", created by former marine biologist and animator Stephen Hillenburg, aired
on Nickelodeon from May 6, 2005 to July 24, 2007, and contained 20 episodes, beginning with the episodes "Fear of a Krabby Patty" and "Shell of a Man".

 The series chronicles the exploits and adventures of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The season was executive produced by series creator Hillenburg, while writer Paul Tibbitt acted as the supervising/co-executive producer and showrunner.

 The show underwent a hiatus on television as Hillenburg halted the production in 2002 to work on the film adaptation of the series, "The SpongeBob SquarePants
Movie".

 Once the film was finalized and the previous season had completed broadcast on television, Hillenburg wanted to end the show, but the success of the series led
to more episodes, so Tibbitt took over Hillenburg's position as showrunner and began working on a fourth season for broadcast in 2005.
</item 8>

<item 9>
SpongeBob SquarePants
SpongeBob SquarePants is an American animated television series created by marine biologist and animator Stephen Hillenburg for Nickelodeon.

 The series chronicles the adventures and endeavors of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The series' popularity has made it a media franchise, as well as the highest rated series to ever air on Nickelodeon, and the most distributed property of MTV
Networks.

 As of 2015, the media franchise has generated $12 billion in merchandising revenue for Nickelodeon.
</item 9>

<item 10>
Legends of Bikini Bottom
Legends of Bikini Bottom is an anthology series of six episodes in the American animated television series "SpongeBob SquarePants", as part of its seventh
season.

 As the name suggests, the episodes have plots involving things like monsters, magic and mythical creatures.

 The first episode called "Trenchbillies" first premiered on the online social networking service Facebook on January 27, 2011 before airing on Nickelodeon on
January 29, 2011.

 The four other episodes titled "Sponge-Cano!"

, "The Main Drain", "The Monster Who Came to Bikini Bottom" and "Welcome to the Bikini Bottom Triangle" premiered on Nickelodeon as part of an hour-long special
on January 28, 2011.

 The decision to air the series online was aimed at attracting "the young and the restless hooked to the internet and the social media."
</item 10>
=== END ITEM id=a60379d94f3b5986 ===

=== ITEM id=97d4f6f7b193c9b0 ===
Gold sentences and facts:
Sentence 0: Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.
  Fact 0.0: Pulandian District is one of the seven districts under the administration of Dalian.
  Fact 0.1: Pulandian District is located in the south of Liaoning province.
  Fact 0.2: Liaoning province is in the People's Republic of China.
Sentence 1: Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.
  Fact 1.0: Kaiyuan is a county-level city.
  Fact 1.1: Kaiyuan is in the northeast of Liaoning.
  Fact 1.2: Liaoning is in the People's Republic of China.
  Fact 1.3: Kaiyuan borders Jilin for a small section to the north.

Retrieved items:
<item 1>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 1>

<item 2>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 2>

<item 3>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 3>

<item 4>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
</item 4>

<item 5>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 5>

<item 6>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 6>

<item 7>
Kaiyuan West Railway Station
Kaiyuan West Railway Station is a railway station of Hada Passenger Railway and located in Liaoning, China
</item 7>

<item 8>
Taiyang Subdistrict
Taiyang Subdistrict () is a subdistrict of Wafangdian, Dalian District, Liaoning, China.

 It is based around the small town of Taiyang, Taiyangsheng or Taiyangshengxiang, which lies 19 kilometres east by road of Fuzhou, 64 kilometres north by road
from Pulandian.

 To the east is the DFM reservoir.
</item 8>

<item 9>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 9>

<item 10>
Shuangta, Pulandian
Shuangta () is a town under the administration of Pulandian City in southern Liaoning province, China, located 58 km northeast of downtown Pulandian and about
120 km northeast of Dalian.

 , it has 9 villages under its administration.
</item 10>
=== END ITEM id=97d4f6f7b193c9b0 ===

=== ITEM id=15e701b556c5aae3 ===
Gold sentences and facts:
Sentence 0: Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming pieces and a game system.
  Fact 0.0: Icehouse pieces are also called Icehouse Pyramids.
  Fact 0.1: Treehouse pieces are also called Treehouse Pyramids.
  Fact 0.2: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, and Treehouse Pyramids are officially called Looney Pyramids.
  Fact 0.3: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are nestable pyramid-shaped gaming pieces.
  Fact 0.4: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are stackable pyramid-shaped gaming pieces.
  Fact 0.5: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are a game system.
Sentence 1:  The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
  Fact 1.0: The Looney Pyramids game system was invented by Andrew Looney.
  Fact 1.1: The Looney Pyramids game system was invented by John Cooper.
  Fact 1.2: The Looney Pyramids game system was invented in 1987.
  Fact 1.3: The Looney Pyramids game system was originally for use in the game of Icehouse.
Sentence 2:  In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
  Fact 2.0: "Kill Doctor Lucky" won the Origins Award in 1998.
  Fact 2.1: "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".

Retrieved items:
<item 1>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 1>

<item 2>
Icehouse (song)
"Icehouse" is a song by the Australian rock band Flowers.

 It was released as a single in Europe in 1982 by Chrysalis Records from the band's first album, "Icehouse", after the band changed its name to Icehouse.

 In the United States, the song peaked at number 28 on the "Billboard" Top Tracks chart in 1981.
</item 2>

<item 3>
The Hunter (EP)
The Hunter is the first EP (and follow up to the 2010 solo album "The Boxer") by Kele Okereke (under the professional name Kele), lead singer and rhythm
guitarist of the British rock band Bloc Party.

 It was released on 7 November 2011 by Wichita Recordings in the UK, set back a week from the original release date, and 3 days earlier on 4 November 2011 by
Wichita Recordings and Liberator Music in Australia.

 The first single released from the EP was "What Did I Do?

", which features guest vocals from Lucy Taylor.

 The music video was released on 13 September 2011, with the song released as a digital single in Japan 10 days later.
</item 3>

<item 4>
Can't Help Myself
"Can't Help Myself" is the first single released by the Australian synthpop/rock band Flowers, later known as Icehouse.

 It was released in May 1980 as a 7" vinyl single on independent label, Regular Records, five months ahead of debut album "Icehouse".

 A 10" vinyl single was released in July and had a cover depicting individual images of band members diagonally across the band's name and the single's title
(see infobox at right middle).

 It peaked at #10 on the Australian Singles Charts.
</item 4>

<item 5>
Icehouse pieces
Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming
pieces and a game system.

 The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
</item 5>

<item 6>
We Can Get Together
"We Can Get Together" is the second single released by the Australian rock band Flowers, later known as Icehouse.

 It was released in October 1980, on the independent label Regular Records from their first album, "Icehouse", two weeks before the album itself was released.

 It peaked at #16 on the Australian Kent Music Report Singles Charts.
</item 6>

<item 7>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 7>

<item 8>
Boxes (Icehouse album)
Boxes is a soundtrack album by Australian band Icehouse, released by Festival Records / Chrysalis Records in November 1985.

 The work was originally conceived by its composers, Iva Davies and Robert Kretschmer, in collaboration with choreographer Graeme Murphy of the Sydney Dance
Company for performance as the ballet "Boxes".

 The first live performance of "Boxes" was given by the Sydney Dance Company together with Davies and Kretschmer of Icehouse and guest percussionist Masaki
Tanazawa in the Opera Theatre of the Sydney Opera House on 7 November 1985.

 "Boxes" was released in the US under the name Sydney Dance Company.

 The single "No Promises" was released as a 12" in November and a cassingle in December on Regular Records for Australian and New Zealand markets, it contained
other tracks from "Boxes"; two tracks from "Boxes" were later released on their next studio album "Measure for Measure" in April 1986.
</item 8>

<item 9>
No Promises (Icehouse song)
"No Promises" is the first single released by Australian band, Icehouse from the band's 1986 album, "Measure for Measure" and was released in November 1985 on
Regular Records as 7" Vinyl Single and 12" Vinyl Single formats.

 It was also released in the UK and Europe by Chrysalis Records again on 7" and 12" formats but with different track listings.

 It was subsequently released in the US by Chrysalis on 7" and 12" formats, again with different track listings.

 It peaked at #30 on the Australian singles charts.

 A remix version by [love] tatto was released on the Icehouse album "Meltdown" in 2002.
</item 9>

<item 10>
Hidden Expedition
Hidden Expedition is a series of single player hidden object casual games developed by the internal studios of Big Fish Games for the first five installments
(using Big Fish Games Framework as the engine, and with the help of Flood Light Games in the 5th game), and by Eipix Entertainment for all subsequent
installments.

 As of September 2017, a total of fifteen games in the series have been released.

 The Hidden Expedition series marks the second major hidden object game brand from Big Fish Games.

 The second game in the series, "Hidden Expedition: Everest", would go on to be the first game Big Fish released to the iPhone.
</item 10>
=== END ITEM id=15e701b556c5aae3 ===

=== ITEM id=0e2fe13ea7d18623 ===
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
Office of the First Lady of the United States
The Office of the First Lady of the United States is the staff accountable to the First Lady of the United States.

 The office and its responsibilities, while not mandated, have grown as the role of the First Lady has grown and formalized through the history of the United
States.

 The Office of the First Lady is an entity of the White House Office, part of the Executive Office of the President.
</item 1>

<item 2>
Mary Elizabeth Bliss
Mary Elizabeth "Betty" Taylor Bliss Dandridge, born Mary Elizabeth Taylor (April 20, 1824 – July 25, 1909), was the youngest of the three surviving daughters of
President Zachary Taylor (1849-1850) and Margaret Mackall Smith Taylor.
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
Vicki Risch
Vicki Risch is the former First Lady of Idaho and the wife of U.S. Sen. Jim Risch, who served as Governor of Idaho in 2006.

 She became First Lady on May 26, 2006, when her husband succeeded former Gov. Dirk Kempthorne, who resigned to become United States Secretary of the Interior.

 Mrs. Risch succeeded former First Lady Patricia Kempthorne who had held the post for over seven years.

 Mrs. Risch served as First Lady until January 2007, as her husband did not seek a full term as governor, but rather was reelected to his old post as lieutenant
governor.
</item 5>

<item 6>
Melania Trump
Melania Trump ( ; born Melanija Knavs; ] , Germanized to Melania Knauss; April 26, 1970) is the current First Lady of the United States.

 Before marrying future President Donald Trump in 2005, she worked as a fashion model; by 2016 she considered herself a "full-time mom."
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
Abigail Adams
Abigail Adams ("née" Smith; November 22  1744 – October 28, 1818) was the closest advisor and wife of John Adams, as well as the mother of John Quincy Adams.

 She is sometimes considered to have been a Founder of the United States, and is now designated as the first Second Lady and second First Lady of the United
States, although these titles were not in use at the time.
</item 8>

<item 9>
Louisa Adams
Louisa Catherine Johnson Adams (February 12, 1775 – May 15, 1852), wife of John Quincy Adams, was First Lady of the United States from 1825 to 1829.

 The daughter of the American Consul in London, she was the first First Lady to be born outside the United States, or outside areas that were later to become
part of the United States, such as the Thirteen Colonies – a distinction that would not be replicated until 192 years later by Melania Trump.
</item 9>

<item 10>
Anna Harrison
Anna Tuthill Symmes Harrison (July 25, 1775 – February 25, 1864), wife of President William Henry Harrison and grandmother of President Benjamin Harrison, was
nominally First Lady of the United States during her husband's one-month term in 1841, but she never entered the White House.

 At the age of 65 years during her husband's presidential term, she is the oldest woman ever to become First Lady, as well as having the distinction of holding
the title for the shortest length of time, and the first person to be widowed while holding the title.

 She was the last First Lady to have been born in British America.
</item 10>
=== END ITEM id=0e2fe13ea7d18623 ===

=== ITEM id=7299a6a77d64cbce ===
Gold sentences and facts:
Sentence 0: Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
  Fact 0.0: Robert Paul Holdstock was born on 2 August 1948.
  Fact 0.1: Robert Paul Holdstock died on 29 November 2009.
  Fact 0.2: Robert Paul Holdstock was an English novelist.
  Fact 0.3: Robert Paul Holdstock was an English author.
  Fact 0.4: Robert Paul Holdstock was best known for his works of Celtic fantasy literature.
  Fact 0.5: Robert Paul Holdstock was best known for his works of Nordic fantasy literature.
  Fact 0.6: Robert Paul Holdstock was best known for his works of Gothic fantasy literature.
  Fact 0.7: Robert Paul Holdstock was best known for his works of Pictish fantasy literature.
  Fact 0.8: Robert Paul Holdstock's works are predominantly in the fantasy subgenre of mythic fiction.
Sentence 1: Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.
  Fact 1.0: Henry Louis Mencken was born on September 12, 1880.
  Fact 1.1: Henry Louis Mencken died on January 29, 1956.
  Fact 1.2: Henry Louis Mencken was an American journalist.
  Fact 1.3: Henry Louis Mencken was an American satirist.
  Fact 1.4: Henry Louis Mencken was an American cultural critic.
  Fact 1.5: Henry Louis Mencken was a scholar of American English.

Retrieved items:
<item 1>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 1>

<item 2>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 2>

<item 3>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 3>

<item 4>
Edgar Allan Poe’s humorous “A Predicament,” originally “The Scythe of Time,” parodies Gothic sensation tales and is often paired with “How to Write a Blackwood
Article.” British writer Michael Moorcock (born 1939), famed for Elric, wrote the science-fiction novels *Behold the Man* (expanded from a 1966 novella; title
from John’s Gospel), *The Final Programme*, *The Warlord of the Air* (1971; an
</item 4>

<item 5>
Robert Holdstock
Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish
fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
</item 5>

<item 6>
The entries profile notable figures across philanthropy, arts, journalism, religion, academia, business, and politics: Edmund R. Harris funded Preston
institutions; Edmund de Unger built the Keir Islamic-art collection, now loaned to Dallas; Mencken was a major American satirist; and H. Donald Wilson helped
create LexisNexis. Others include leaders, scholars, artists, and advocates such as George Albert Smith, Andrew Benjamin, Inderpal Grewal, Emeka Es
</item 6>

<item 7>
The entries profile writers, artists, scholars, and public figures across literature, journalism, film, and politics. Highlights include Thomas Merton, the
Catholic monk and writer honored by a peace award and studied by an international society; W. Somerset Maugham, whose namesake prize supports young writers’
travel; and notable authors such as Mencken, Orwell, Updike, Malamud, Chatwin, and Gass. Other topics include literary awards, bibliographies, influential
editors, Camb
</item 7>

<item 8>
Robert Holdstock bibliography
This is a bibliography of fantasy author Robert Holdstock.
</item 8>

<item 9>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 9>

<item 10>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 10>
=== END ITEM id=7299a6a77d64cbce ===

=== ITEM id=f033f1a7cdeca2ec ===
Gold sentences and facts:
Sentence 0: Erving Goffman (11 June 1922 – 19 November 1982) was a Canadian-American sociologist and writer, considered by some "the most influential American sociologist of the twentieth century".
  Fact 0.0: Erving Goffman was born on 11 June 1922.
  Fact 0.1: Erving Goffman died on 19 November 1982.
  Fact 0.2: Erving Goffman was a Canadian-American sociologist.
  Fact 0.3: Erving Goffman was a writer.
  Fact 0.4: Some consider Erving Goffman the most influential American sociologist of the twentieth century.
Sentence 1:  In 2007 he was listed by "The Times Higher Education Guide" as the sixth most-cited author in the humanities and social sciences, behind Anthony Giddens, Pierre Bourdieu and Michel Foucault, and ahead of Jürgen Habermas.
  Fact 1.0: In 2007 Erving Goffman was listed by The Times Higher Education Guide.
  Fact 1.1: The Times Higher Education Guide listed Goffman as the sixth most-cited author in the humanities and social sciences in 2007.
  Fact 1.2: In that 2007 listing, Anthony Giddens, Pierre Bourdieu and Michel Foucault were ranked ahead of Goffman.
  Fact 1.3: In that 2007 listing, Goffman was ranked ahead of Jürgen Habermas.
Sentence 2: Paul-Michel Foucault (15 October 1926 – 25 June 1984), generally known as Michel Foucault (] ), was a French philosopher, historian of ideas, social theorist, and literary critic.
  Fact 2.0: Paul-Michel Foucault was born on 15 October 1926.
  Fact 2.1: Paul-Michel Foucault died on 25 June 1984.
  Fact 2.2: Paul-Michel Foucault is generally known as Michel Foucault.
  Fact 2.3: Michel Foucault was a French philosopher.
  Fact 2.4: Michel Foucault was a historian of ideas.
  Fact 2.5: Michel Foucault was a social theorist.
  Fact 2.6: Michel Foucault was a literary critic.

Retrieved items:
<item 1>
Robert Calder (writer)
Robert Lorin Calder, a Canadian writer and professor, won the Governor General's Award for English-language non-fiction in 1989 for his "Willie: The Life of W.
Somerset Maugham", a biography based on extensive archival work and interviews with surviving associates of Maugham, in particular Alan Searle.

 Unlike Ted Morgan, who had obtained permission from Maugham's executors to publish from Maugham's letters in his biography (1980), Calder was refused
permission to do so by the Royal Literary Fund and had to rely on paraphrase in referencing Maugham's unpublished correspondence.
</item 1>

<item 2>
Ray Birdwhistell
Collaborations with others, including initially Margaret Mead and Gregory Bateson, and later, Erving Goffman and Dell Hymes had huge influence on his work.

 For example, the book he is best known for, "Kinesics and Context", "would not have appeared if it had not been envisaged by Erving Goffman" and he explicitly
stated "the paramount and sustaining influence upon my work has been that of anthropological linguistics", a tradition most directly represented at the
University of Pennsylvania by Hymes.
</item 2>

<item 3>
Erving Goffman
Erving Goffman (11 June 1922 – 19 November 1982) was a Canadian-American sociologist and writer, considered by some "the most influential American sociologist
of the twentieth century".

 In 2007 he was listed by "The Times Higher Education Guide" as the sixth most-cited author in the humanities and social sciences, behind Anthony Giddens,
Pierre Bourdieu and Michel Foucault, and ahead of Jürgen Habermas.
</item 3>

<item 4>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 4>

<item 5>
Dramaturgy (sociology)
Dramaturgy is a sociological perspective commonly used in microsociological accounts of social interaction in everyday life.

 The term was first adapted into sociology from the theatre by Erving Goffman, who developed most of the related terminology and ideas in his 1959 book, "The
Presentation of Self in Everyday Life".

 Kenneth Burke, whom Goffman would later acknowledge as an influence, had earlier presented his notions of dramatism in 1945, which in turn derives from
Shakespeare.

 However, the fundamental difference between Burke's and Goffman's view is that Burke believed that life was in fact theatre, whereas Goffman viewed theatre as
a metaphor.

 If we imagine ourselves as directors observing what goes on in the theatre of everyday life, we are doing what Goffman called dramaturgical analysis, the study
of social interaction in terms of theatrical performance.
</item 5>

<item 6>
George Orwell bibliography
The bibliography of George Orwell includes journalism, essays, novels and non-fiction books written by the British writer Eric Blair (1903–50), either under his
own name or, more usually, under his pen name George Orwell.

 Orwell was a prolific writer on topics related to contemporary English society and literary criticism, whom the British newsweekly "The Economist" in 2008
declared "perhaps the 20th century's best chronicler of English culture."

 His non-fiction cultural and political criticism constitutes the majority of his work, but Orwell also wrote in several genres of fictional literature.
</item 6>

<item 7>
The Presentation of Self in Everyday Life
The Presentation of Self in Everyday Life is a 1956 sociology book by Erving Goffman, in which Goffman uses the imagery of the theatre in order to portray the
importance of human social interaction.

 Originally published in Scotland in 1956 and in the United States in 1959, it is Goffman’s first and most famous book, for which he received the American
Sociological Association’s MacIver award in 1961.
</item 7>

<item 8>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 8>

<item 9>
Goffman (disambiguation)
Erving Goffman (1922–1982) was a Canadian sociologist
</item 9>

<item 10>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 10>
=== END ITEM id=f033f1a7cdeca2ec ===

=== ITEM id=d841090926d888ff ===
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
=== END ITEM id=d841090926d888ff ===

=== ITEM id=fb16faeb55509d72 ===
Gold sentences and facts:
Sentence 0: Pramod Bhasin stepped down as President and CEO of Genpact, India's largest business process outsourcing (BPO) company, becoming non-executive Vice Chairman in 2011.
  Fact 0.0: Pramod Bhasin stepped down as President of Genpact.
  Fact 0.1: Pramod Bhasin stepped down as CEO of Genpact.
  Fact 0.2: Genpact is India's largest business process outsourcing (BPO) company.
  Fact 0.3: Pramod Bhasin became non-executive Vice Chairman in 2011.
Sentence 1: Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.
  Fact 1.0: Genpact is a global business process management and services corporation.
  Fact 1.1: Genpact is an IT corporation.
  Fact 1.2: Genpact has key offices in New York City, United States.

Retrieved items:
<item 1>
Alex Algard
Alex Algard is an Internet entrepreneur.

 He is the founder and former CEO of CarDomain, the founder and former CEO of Whitepages.com, and the founder and current CEO of Hiya.
</item 1>

<item 2>
Mike Long (American businessman)
Mike Long is an American business man, former CEO of several public companies, and currently a founding partner of Sulgrave Partners LLC.

 He served as the President and Chief Executive Officer of Continuum, an Austin, Texas IT consulting company, from 1991 to 1997, having started with Continuum
as a Director in 1983.

 In 1997, Long was named CEO of Healtheon Corporation (now WebMD), succeeding former CEO David Schnell.

 Long oversaw Healtheon's initial public offering, traveling between Europe and the United States to woo investors.

 Long was able to secure the required investment funds, and saw Healtheon's stock price rise from $8 to a high of $120.

 In 2002, Long was recruited to fix the financial struggles of Move, Inc., a company plagued by more than $4 billion in lawsuits and hemorrhaging tens of
millions of dollars a quarter.

 As Chief Executive Officer, Long was able to revive Homestore, Inc., by changing the business model, rebranding the company as Move, Inc. and returning it to
profitability.
</item 2>

<item 3>
Genpact
Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.

 It’s listed on the New York Stock Exchange under the symbol G.In 2016, the company reported net revenues of US$2.57 billion with more than 77,000 employees in
20 countries.

 Genpact provides consulting services to hundreds of clients, including approximately one-fifth of the Fortune Global 500.
</item 3>

<item 4>
Pramod Bhasin
Pramod Bhasin stepped down as President and CEO of Genpact, India's largest business process outsourcing (BPO) company, becoming non-executive Vice Chairman in
2011.
</item 4>

<item 5>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 5>

<item 6>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 6>

<item 7>
Prem Ganapathy
Prem Ganapathy is an Indian entrepreneur and businessman.

 He is the founder of the restaurant chain Dosa plaza.

 Starting with meager investment, he expanded Dosa plaza into a restaurant chain with 45 outlets in India, New Zealand, Oman, and UAE.
</item 7>

<item 8>
Richard Yoo
Richard Yoo is an American entrepreneur, the co-founder and former CEO of the web hosting company Rackspace, and the founder and former CEO of web hosting
company ServerBeach.
</item 8>

<item 9>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 9>

<item 10>
Mike Long (American businessman)
and returning it to profitability.

 Touching on his experience of bringing about the initial public offerings of web-based businesses, Long would say that investors needed to be presented "with
an entirely new face every few months," and that "the only way to run one of these Silicon Valley companies was to forget everything you'd learned outside of
Silicon Valley."
</item 10>
=== END ITEM id=fb16faeb55509d72 ===

=== ITEM id=7fbd4a7c5c23f96a ===
Gold sentences and facts:
Sentence 0: Brett Dallas McClure (born February 19, 1981 in Yakima, Washington) is a retired American gymnast.
  Fact 0.0: Brett Dallas McClure was born on February 19, 1981.
  Fact 0.1: Brett Dallas McClure was born in Yakima, Washington.
  Fact 0.2: Brett Dallas McClure is a retired American gymnast.
Sentence 1:  He won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia, and later helped his U.S. gymnastics team earn a silver in the team competition at the 2004 Summer Olympics in Athens.
  Fact 1.0: Brett Dallas McClure won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia.
  Fact 1.1: Brett Dallas McClure later helped his U.S. gymnastics team earn a silver medal in the team competition at the 2004 Summer Olympics in Athens.
Sentence 2:  The competition was held in Brisbane, Queensland, Australia from 29 August to 9 September 2001.
  Fact 2.0: The competition was held in Brisbane, Queensland, Australia.
  Fact 2.1: The competition was held from 29 August to 9 September 2001.

Retrieved items:
<item 1>
2001 Goodwill Games
The 2001 Goodwill Games was the fifth and final edition of the international multi-sport event.

 The competition was held in Brisbane, Queensland, Australia from 29 August to 9 September 2001.

 A total around 1300 athletes took part in 14 sporting competitions.
</item 1>

<item 2>
Tabcorp Park Menangle
Tabcorp Park Menangle is a harness racing track in Sydney.

 Their main races are the Miracle Mile Pace, the NSW Derby, the NSW Oaks, the Len Smith Mile and the qualifiers for the Miracle Mile, the Allied Express Sprint
and the Canadian Club Sprint.

 It replaced Harold Park as Sydney's premier harness track in 2010.

 Fred Hastings and Anthony Manton are two callers that have called at the track before.
</item 2>

<item 3>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 3>

<item 4>
2002 Commonwealth Games
The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25
July to 4 August 2002.

 The 2002 Games were to be hosted in the United Kingdom to coincide with the Golden Jubilee of Elizabeth II, head of the Commonwealth, and Manchester was
selected for the 2002 Games ahead of London.

 The XVII Commonwealth Games was, prior to the 2012 Summer Olympics, the largest multi-sport event ever to be held in the UK, eclipsing the London 1948 Summer
Olympics in numbers of teams and athletes participating.

 In terms of sports and events, the 2002 Games were the largest Commonwealth Games in history featuring 281 events across 17 sports.
</item 4>

<item 5>
Brett McClure
Brett Dallas McClure (born February 19, 1981 in Yakima, Washington) is a retired American gymnast.

 He won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia, and later helped his U.S. gymnastics team earn a silver in the
team competition at the 2004 Summer Olympics in Athens.

 During his sporting career, McClure has collected two more silver medals in the same program at the World Artistic Gymnastics Championships (2001 and 2003).

 McClure is currently the High Performance Director with USA men's gymnastics.
</item 5>

<item 6>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 6>

<item 7>
Klete Keller
Klete D. Keller (born March 21, 1982) is an American former competition swimmer who won medals at the 2000 Summer Olympics and the 2004 Summer Olympics in the
400-meter freestyle and the 4×200-meter freestyle relay.

 In the 4×200-meter freestyle relay, Keller held off a charging Ian Thorpe in the anchor leg to win the race by 0.13 seconds.

 This was the first time Australia had been beaten in the event in over seven years.

 The American relay of Michael Phelps, Ryan Lochte, Peter Vanderkaay, and Keller are undefeated since the Athens games.

 Vanderkaay, Larsen Jensen, Erik Vendt, and Keller make up the core of the premier American mid-distance/distance freestyle swimmers.
</item 7>

<item 8>
Brian MacLaren
Brian MacLaren (born 21 December 1943) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1972 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Bill Crothers, Ross MacKenzie, and Robert McLaren) and third in the 1967 Pan
American Games 800 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Ross MacKenzie and Bill Crothers.
</item 8>

<item 9>
Offshore Festival
The Offshore Festival was a camp-out rock and alternative music festival held during Easter at a farm near Torquay, Victoria, Australia from the late 1990s to
2001.

 It was run by the same organisers as the Falls Festival, held at nearby Lorne on New Year's Eve.

 It interlinked with the Rip Curl Pro surfing event, held at nearby Bells Beach, and festival ticketholders had free admission to the surfing event.
</item 9>

<item 10>
Anthony Biddle (Paralympian)
Anthony John Biddle, OAM (born 18 June 1975) is an Australian Paralympic tandem cyclist and athlete.

 He was born in the New South Wales city of Gosford.

 He competed in athletics without winning any medals at the 1996 Atlanta Games and the 2000 Sydney Games.

 At the 2004 Athens Games, he switched to cycling and won a gold medal in the Men's 1 km Time Trial Tandem B1–3 event, for which he received a Medal of the
Order of Australia, and a bronze medal in the Men's Sprint Tandem B1–3 event.

 Kial Stewart was his pilot for both events.
</item 10>
=== END ITEM id=7fbd4a7c5c23f96a ===

=== ITEM id=7083f2abc51903d0 ===
Gold sentences and facts:
Sentence 0: The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil War.
  Fact 0.0: The Seven Days Battles were a series of six major battles.
  Fact 0.1: The Seven Days Battles were fought over the seven days from June 25 to July 1, 1862.
  Fact 0.2: The Seven Days Battles were fought near Richmond, Virginia.
  Fact 0.3: The Seven Days Battles took place during the American Civil War.
Sentence 1: The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.
  Fact 1.0: The Battle of Manila took place from February 3, 1945 to March 3, 1945.
  Fact 1.1: The Battle of Manila was a major battle of the Philippine campaign of 1944-45.
  Fact 1.2: The Battle of Manila took place during the Second World War.

Retrieved items:
<item 1>
Battle of Bataan
The Battle of Bataan (7 January – 9 April 1942) represented the most intense phase of Imperial Japan's invasion of the Philippines during World War II.

 In January 1942, forces of the Imperial Japanese Army and Navy invaded Luzon along with several islands in the Philippine Archipelago after the bombing of the
American naval base at Pearl Harbor.

 The commander-in-chief of all Filipino and American forces in the islands, General Douglas MacArthur, consolidated all of his Luzon-based units on the Bataan
Peninsula to fight against the Japanese invaders.

 By this time, the Japanese controlled nearly all of Southeast Asia.

 The Bataan peninsula and the island of Corregidor were the only remaining Allied strongholds in the region.

 Despite a lack of supplies, Filipino and American forces managed to fight the Japanese for three months, engaging them initially in a fighting retreat
southward.

 As the combined Filipino and American forces made a last stand, the delay cost the Japanese valuable time and prevented immediate victory across the Pacific.
</item 1>

<item 2>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 2>

<item 3>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 3>

<item 4>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 4>

<item 5>
Battle of Manila (1574)
The Battle of Manilla (1574) was a battle in the Manila area mainly in the location of what is now Parañaque between Chinese pirates, led by Limahong and the
Spanish colonial forces and their native allies.

 The battle occurred on November 29, 1574 when Limahong's fleet landed in the town of Parañaque and from there, began to assault the fortifications of
Intramuros.

 Initially, the inhabitants where disorganized and Limahong's forces routed them.

 Furthermore, the Chinese killed the Master-of-Camp of the Spanish, Martin De Goiti.

 This caused them to delay their assault on Manila as Martin de Goiti's house was an obstacle in their march.

 However, upon the arrival of a certain Filipino hero called, Galo, resistance started to organize.

 Under Galo's command, they were able to resist China-born Limahong until Mexico-born Juan de Salcedo arrived from Ilocos with 300 Ilocano Warriors and Limahong
was defeated and eventually forced to retreat.

 Thereafter Limahong abandoned his plans to invade Manila and instead, set up a temporary kingdom in Pangasinan.
</item 5>

<item 6>
Seven Days Confederate order of battle
The following Confederate States Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization during the battle, the casualty returns and the reports.

 The Union order of battle is listed separately.
</item 6>

<item 7>
Battle of Manila (1945)
The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.

 It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.

 The month-long battle, which resulted in the death of over 100,000 civilians and the complete devastation of the city, was the scene of the worst urban
fighting in the Pacific theater.

 Japanese forces committed mass murder against Filipino civilians during the battle.

 Along with massive loss of life, the battle also destroyed architectural and cultural heritage dating back to the city's foundation.

 The battle ended the almost three years of Japanese military occupation in the Philippines (1942–1945).

 The city's capture was marked as General Douglas MacArthur's key to victory in the campaign of reconquest.
</item 7>

<item 8>
Seven Days Union order of battle
The following Union Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization-return of casualties during the battle and the reports.

 The Confederate order of battle is listed separately.
</item 8>

<item 9>
Battle of Bataan
The surrender at Bataan, with 76,000 soldiers surrendering in the Philippines altogether, was the largest in American and Filipino military histories, and was
the largest United States surrender since the American Civil War's Battle of Harper's Ferry.

 Soon afterwards, Filipino and American prisoners of war were forced into the Bataan Death March.
</item 9>

<item 10>
Seven Days Battles
The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil
War.

 Confederate General Robert E. Lee drove the invading Union Army of the Potomac, commanded by Maj. Gen. George B. McClellan, away from Richmond and into a
retreat down the Virginia Peninsula.

 The series of battles is sometimes known erroneously as the Seven Days Campaign, but it was actually the culmination of the Peninsula Campaign, not a separate
campaign in its own right.
</item 10>
=== END ITEM id=7083f2abc51903d0 ===

=== ITEM id=f1cbbc0ecc050d8b ===
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
Barry Rose
Barry Michael Rose OBE, FRAM, FRSCM, Hon.D.Mus, Hon.FGCM, Hon FRCO, M.Univ., (Surrey), born 24 May 1934) is a choir trainer and organist.

 He is best known for founding the choir and the pattern of daily sung Worship at the new Guildford Cathedral in 1961, and later, on 29 July 1981, directing the
music at the wedding of Charles, Prince of Wales and Diana, Princess of Wales formerly Lady Diana Frances Spencer, at St Paul's Cathedral in London.
</item 1>

<item 2>
Lanesborough School
Lanesborough School is an independent, preparatory school in Guildford, Surrey.

 The school was established in 1930 and acts as the choir school for Guildford Cathedral.

 Lanesborough is an important feeder school for RGS Guildford.
</item 2>

<item 3>
Guildford Cathedral
The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.

 The Lord Onslow donated the land on which the Cathedral was built with a covenant for this purpose.

 Designed by Sir Edward Maufe and built between 1936 and 1961, it is the seat of the Diocese of Guildford.
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
Exterior sculpture of Guildford Cathedral
The Exterior sculpture of Guildford Cathedral provides many artistic features, including sculptures, engravings and more by some of England’s finest sculptors
and craftsmen of the 1950s and 1960s.

 The people who worked on the cathedral include: Edward Maufe, Alan Collins, Vernon Hill, Eric Gill, John Hutton, Dennis Huntley and others.
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
Dean of Guildford
The Dean of Guildford is the head ("primus inter pares" – first among equals) and chair of the chapter of canons, the ruling body of Guildford Cathedral.

 The dean and chapter are based at the "Cathedral Church of the Holy Spirit" in Guildford.

 The cathedral is the mother church of the Diocese of Guildford and seat of the Bishop of Guildford.

 The current dean is Dianna Gwilliams, who was installed on 15 September 2013.
</item 7>

<item 8>
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
</item 8>

<item 9>
Diocese of Guildford
The Diocese of Guildford is a Church of England diocese covering nine of the eleven districts in Surrey, much of north-east Hampshire and a parish in Greater
London.

 The cathedral is Guildford Cathedral and the bishop is the Bishop of Guildford.

 Of the two provinces of the church it falls within the Province of Canterbury.
</item 9>

<item 10>
St Paul's Cathedral, Rockhampton
St Paul's Anglican Cathedral is a heritage-listed cathedral at 89 William Street, Rockhampton, Rockhampton Region, Queensland, Australia.

 It was designed principally by Annersley Voysey and built from 1883 to 1953.

 It is also known as St Paul's Church of England and St Paul's Church of England Cathedral.

 It was added to the Queensland Heritage Register on 21 October 1992.

 The associated hall and offices were listed on the Queensland Heritage Register on 23 June 2000.
</item 10>
=== END ITEM id=f1cbbc0ecc050d8b ===
