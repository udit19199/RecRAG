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

=== ITEM id=d9ebaf63473a89ce ===
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
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 1>

<item 2>
Dan Clark
Daniel Gregory Clark (born 3 July 1976 in Bromley, London) is an English actor, comedian, writer, director, and singer.

 He is best known for playing Don Danbury on the BBC Three sitcom "How Not to Live Your Life", which he also wrote, co-produced, and sometimes directed.

 He has been a regular on the British comedy scene as both a sketch and stand-up comedian.
</item 2>

<item 3>
Nicholas Mele
Nicholas Mele is an actor who has starred in many movies and on television.

 His first movie role was in the 1976 movie "The Ritz".

 Other movie roles include "Some Kind of Hero" (1982) and "Young Doctors in Love" (1982).

 His most well-known film role was in the 1988 hit horror movie "" as Dennis Johnson.

 A year later, he reprised his role in the hit sequel "".

 His most recent movie is the 2003 film "The Great Gabble".
</item 3>

<item 4>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 4>

<item 5>
Nicholas Hytner
Sir Nicholas Robert Hytner (born 7 May 1956) is an English theatre director, film director, and film producer.

 He was previously the Artistic Director of London's National Theatre.

 His major successes while director include "Miss Saigon", "The History Boys" and "One Man, Two Guvnors".
</item 5>

<item 6>
Robert Oliveri
Robert Dane Oliveri (born April 28, 1978) is a former American actor.

 He is best known for his portrayal of Nick Szalinski in the 1989 Disney film, "Honey, I Shrunk the Kids", for which he was nominated for a Young Artist Award
and a Saturn Award.

 He reprised the role in the 1992 sequel, "Honey, I Blew Up the Kid" and the 3D film, "Honey, I Shrunk the Audience".

 He is also known as Kevin, Winona Ryder's little brother, in Tim Burton's "Edward Scissorhands".
</item 6>

<item 7>
Fayard Nicholas
Fayard Antonio Nicholas (October 20, 1914 – January 24, 2006) was an American choreographer, dancer and actor.

 He and his younger brother Harold Nicholas made up the Nicholas Brothers tap-dance duo, who starred in the MGM musicals "An All-Colored Vaudeville Show"
(1935), "Stormy Weather" (1943), "The Pirate" (1948), and "Hard Four" (2007).

 The Nicholas brothers also starred in the 20th Century-Fox musicals "Down Argentine Way" (1940), "Sun Valley Serenade" (1941), and "Orchestra Wives" (1942).
</item 7>

<item 8>
Nat and Alex Wolff
Nat and Alex Wolff are an American pop music duo from New York, New York, consisting of brothers Nat and Alex Wolff.

 The siblings are known for their work on the Nickelodeon television series "The Naked Brothers Band", which was created and produced by their mother, actress
Polly Draper.

 It was adapted from the mockumentary film that Draper wrote and directed.

 The duo's initial teen pop boy band called "The Naked Brothers Band" was depicted as part of its participation in their Nickelodeon series that aired from 2007
to 2009.

 They have released four full-length albums and have been nominated for several awards.
</item 8>

<item 9>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 9>

<item 10>
Chad Gilbert
Chad Everett Gilbert (born March 9, 1981) is an American musician and record producer.

 He is a founding member of the rock band New Found Glory, for whom he plays lead guitar, composes music and provides backing vocals.

 He was also the lead vocalist for New Found Glory's now-defunct side-project, International Superheroes of Hardcore.

 Additionally, Gilbert was the vocalist for the hardcore punk band Shai Hulud between 1995 and 1998, and he rejoined the band in 2012.
</item 10>
=== END ITEM id=d9ebaf63473a89ce ===

=== ITEM id=a9113b78f0ba10bf ===
Gold sentences and facts:
Sentence 0:  It was the first election for Governor of Maine, taking place after Maine separated from Massachusetts and was recognized as a state on March 15, 1820.
  Fact 0.0: The election was the first election for Governor of Maine.
  Fact 0.1: The election took place after Maine separated from Massachusetts.
  Fact 0.2: Maine was recognized as a state on March 15, 1820.
Sentence 1:  Maine's separation from Massachusetts came as a result of The Missouri Compromise.
  Fact 1.0: Maine's separation from Massachusetts came as a result of The Missouri Compromise.
Sentence 2:  This election saw the virtually unanimous election of William King, the man most chiefly responsible for the push for Maine statehood.
  Fact 2.0: The election saw the virtually unanimous election of William King.
  Fact 2.1: William King was the man most chiefly responsible for the push for Maine statehood.
Sentence 3: William King (February 9, 1768June 17, 1852) was an American merchant, shipbuilder, army officer, and statesman from Bath, Maine.
  Fact 3.0: William King was born February 9, 1768.
  Fact 3.1: William King died June 17, 1852.
  Fact 3.2: William King was an American merchant.
  Fact 3.3: William King was an American shipbuilder.
  Fact 3.4: William King was an American army officer.
  Fact 3.5: William King was an American statesman.
  Fact 3.6: William King was from Bath, Maine.

Retrieved items:
<item 1>
John Bowman (pioneer)
Col. John (Johannes) Bowman (17 December 1738 – May 4, 1784) was an 18th-century American pioneer, colonial militia officer and sheriff, the first appointed in
Lincoln County, Kentucky.

 In 1781 he also presided as a justice of the peace over the first county court held in Kentucky.

 The first county-lieutenant and military governor of Kentucky County during the American Revolutionary War, Col. Bowman also, served in the American
Revolution, many times, second in command to General George Rogers Clark, during the Illinois Campaign, which, at the time, doubled the size of the United
States.
</item 1>

<item 2>
Compromise of 1790
The Compromise of 1790 was a compromise between Alexander Hamilton on the one hand and Thomas Jefferson and James Madison whereby Hamilton won the decision for
the national government to take over and pay the state debts, while Jefferson and Madison obtained the national capital (District of Columbia) for the South.

 The compromise resolved the deadlock in Congress.

 Southerners were blocking the assumption of state debts by the treasury, thereby destroying the Hamiltonian program for building a fiscally strong nation
state.

 Northerners rejected the proposal, much desired by Virginians, to locate the permanent national capital on the Virginia-Maryland border.

 The compromise made possible the passage of the Residence and Funding (Assumption) Acts in July and August 1790.

 Historian Jacob Cooke says it is, "generally regarded as one of the most important bargains in American history, ranking just below the better known Missouri
Compromise and the Compromise of 1850."
</item 2>

<item 3>
Louisa Adams
Louisa Catherine Johnson Adams (February 12, 1775 – May 15, 1852), wife of John Quincy Adams, was First Lady of the United States from 1825 to 1829.

 The daughter of the American Consul in London, she was the first First Lady to be born outside the United States, or outside areas that were later to become
part of the United States, such as the Thirteen Colonies – a distinction that would not be replicated until 192 years later by Melania Trump.
</item 3>

<item 4>
William King (governor)
William King (February 9, 1768June 17, 1852) was an American merchant, shipbuilder, army officer, and statesman from Bath, Maine.

 A proponent of statehood for Maine, he became its first governor when it separated from Massachusetts in 1820.
</item 4>

<item 5>
Pottawatomie massacre
The Pottawatomie massacre occurred during the night of May 24 and the morning of May 25, 1856.

 In reaction to the sacking of Lawrence, Kansas by pro-slavery forces, John Brown and a band of abolitionist settlers—some of them members of the Pottawatomie
Rifles—killed five settlers north of Pottawatomie Creek in Franklin County, Kansas.

 This was one of the many bloody episodes in Kansas preceding the American Civil War, which came to be known collectively as Bleeding Kansas.

 Bleeding Kansas was largely brought about by the Missouri Compromise and the Kansas–Nebraska Act.
</item 5>

<item 6>
Anti-Nebraska movement
The Anti-Nebraska movement was a political alignment in the United States formed in opposition to the Kansas-Nebraska Act of 1854 and to its repeal of the
Missouri Compromise provision forbidding slavery in U.S. territories north of latitude 36° 30' N.

 (At the time, the name "Nebraska" could loosely refer to areas west of the Missouri River.)
</item 6>

<item 7>
Henry Smith Lane
Henry Smith Lane (February 24, 1811 – June 19, 1881) was a United States Representative, Senator, and the 13th Governor of Indiana; he was by design the
shortest-serving Governor of Indiana, having made plans to resign the office should his party take control of the Indiana General Assembly and elect him to the
United States Senate.

 He held that office for only two days, and was known for his opposition to slavery.

 A Whig until the party collapsed, he supported compromise with the south.

 He became an early leader in the Republican Party starting in 1856 serving as the president of the first party convention, delivering its keynote address, and
was influential in the nomination of Abraham Lincoln.

 With the repeal of the Missouri Compromise, he became a full-fledged
</item 7>

<item 8>
Missouri Compromise
The Missouri Compromise is the title generally attached to the legislation passed by the 16th United States Congress on May 8, 1820.

 The measures provided for the admission of Maine as a state along with Missouri as a slave state, thus maintaining the balance of power between North and
South.

 As part of the compromise, slavery was prohibited North of the 36°30′ parallel, excluding Missouri.

 President James Monroe signed the legislation on April 6, 1820.
</item 8>

<item 9>
1st Virginia State Regiment
The 1st Virginia State Regiment was a regiment of Virginia Militia that fought during the American Revolutionary War.

 The regiment was authorized by the General Assembly of the Commonwealth of Virginia in December 1776 as a force of regular troops for the Commonwealth's
defense.

 In 1777, Virginia had difficulty meeting its quota for the regular line of the Continental Army.

 As a result, in July 1777 under the command of Colonel George Gibson, the regiment began a march North to temporarily join the Continental Army in the
Philadelphia Campaign.

 In January 1778, the Virginia General Assembly passed an act directing that the 1st Virginia State Regiment "now in Continental service, be continued in said
service instead of the Ninth Virginia Regiment, made prisoners by the enemy in the Battle of Germantown."

 The regiment camped at Valley Forge in the winter of 1777-78 and at Middlebrook in the winter of 1778-79 and participated in the Battle of Monmouth.

 The regiment remained in the service of the Continental Army until late 1779 when redeployed Virginia.
</item 9>

<item 10>
Maine gubernatorial election, 1820
The 1820 Maine gubernatorial election took place on April 3, 1820.

 It was the first election for Governor of Maine, taking place after Maine separated from Massachusetts and was recognized as a state on March 15, 1820.

 Maine's separation from Massachusetts came as a result of The Missouri Compromise.

 This election saw the virtually unanimous election of William King, the man most chiefly responsible for the push for Maine statehood.

 He had no opponents.
</item 10>
=== END ITEM id=a9113b78f0ba10bf ===

=== ITEM id=6c28f02b2e2ed529 ===
Gold sentences and facts:
Sentence 0:  He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his early work as Knox Overstreet in "Dead Poets Society".
  Fact 0.0: The actor referred to as 'He' is best known for the role of Dan Rydell on "Sports Night."
  Fact 0.1: The actor referred to as 'He' is best known for the role of Will Gardner on "The Good Wife."
  Fact 0.2: The role of Will Gardner on "The Good Wife" earned the actor two Primetime Emmy Award nominations.
  Fact 0.3: The actor referred to as 'He' is best known for his early work as Knox Overstreet in "Dead Poets Society."
Sentence 1: Dead Poets Society is a 1989 American drama film directed by Peter Weir, written by Tom Schulman, and starring Robin Williams.
  Fact 1.0: Dead Poets Society is a 1989 American drama film.
  Fact 1.1: Dead Poets Society was directed by Peter Weir.
  Fact 1.2: Dead Poets Society was written by Tom Schulman.
  Fact 1.3: Dead Poets Society stars Robin Williams.
Sentence 2:  Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher who inspires his students through his teaching of poetry.
  Fact 2.0: Dead Poets Society is set in 1959.
  Fact 2.1: Dead Poets Society is set at the fictional elite conservative Vermont boarding school Welton Academy.
  Fact 2.2: Dead Poets Society tells the story of an English teacher.
  Fact 2.3: The English teacher in Dead Poets Society inspires his students through his teaching of poetry.

Retrieved items:
<item 1>
Dead Poets Society
Dead Poets Society is a 1989 American drama film directed by Peter Weir, written by Tom Schulman, and starring Robin Williams.

 Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher who inspires his students
through his teaching of poetry.
</item 1>

<item 2>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 2>

<item 3>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 3>

<item 4>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 4>

<item 5>
Josh Charles
Joshua Aaron "Josh" Charles (born September 15, 1971) is an American stage, film and television actor.

 He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his
early work as Knox Overstreet in "Dead Poets Society".
</item 5>

<item 6>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 6>

<item 7>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 7>

<item 8>
Reunion (1989 film)
The story is centred on the "enchanted friendship" of two teenagers in 1933 Germany.

 Hans Strauss (Christien Anholt) is the son of a Jewish doctor and Konradin Von Lohenburg (Samuel West) is from an aristocratic family.

 The background is the rise of Nazism.

 Jason Robards plays the older Hans in the 1970s as he prepares to travel to Germany for the first time since the 1930s.

 The film was shot on location in Berlin, New York and Stuttgart.

 "Reunion" was nominated for a Golden Palm at the 1989 Cannes Film Festival.
</item 8>

<item 9>
The Boy with a Thorn in His Side
The Boy with a Thorn in His Side is a 2005 comedy film, produced and directed by Mark Jeavonsa as his feature film debut, and starring Alec Sedgley as Billy
Heinlickburger.

 Its title is almost the same as a song by The Smiths.
</item 9>

<item 10>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 10>
=== END ITEM id=6c28f02b2e2ed529 ===

=== ITEM id=5140e5a1e721917f ===
Gold sentences and facts:
Sentence 0: Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.
  Fact 0.0: Lee Dong-hae was born on October 15, 1986.
  Fact 0.1: Lee Dong-hae's name in Hangul is 이동해.
  Fact 0.2: Lee Dong-hae's name in Hanja is 李東海.
  Fact 0.3: Lee Dong-hae is a South Korean singer.
  Fact 0.4: Lee Dong-hae is a South Korean songwriter.
  Fact 0.5: Lee Dong-hae is a South Korean actor.
Sentence 1: Costadinos Contostavlos (born 11 June 1987), better known by his stage name Dappy, is an English singer, songwriter, rapper, and actor.
  Fact 1.0: Costadinos Contostavlos was born on 11 June 1987.
  Fact 1.1: Costadinos Contostavlos is better known by his stage name Dappy.
  Fact 1.2: Costadinos Contostavlos is an English singer.
  Fact 1.3: Costadinos Contostavlos is an English songwriter.
  Fact 1.4: Costadinos Contostavlos is an English rapper.
  Fact 1.5: Costadinos Contostavlos is an English actor.

Retrieved items:
<item 1>
Dappy
Costadinos Contostavlos (born 11 June 1987), better known by his stage name Dappy, is an English singer, songwriter, rapper, and actor.

 He is best known for being the lead singer of the Camden-based grime trio N-Dubz, with his cousin Tulisa, and Fazer.

 He was known for his slightly unusual dress sense and his love for what he describes as "eye-catching headwear".

 His trademark was wearing a selection of woolly chullos (sometimes referred to as "Dappy hats"), turning up one or both of the ear-flaps of the hat.
</item 1>

<item 2>
Ms Panda and Mr Hedgehog
Panda and Hedgehog () is a 2012 South Korean romantic comedy television series starring Lee Donghae of Super Junior and Yoon Seung-ah.

 Produced by Song Hae-sung's TV production venture Lion Fish, it aired on Channel A from August 18 to October 7, 2012 on Saturdays and Sundays at 21:55 for 16
episodes.
</item 2>

<item 3>
Lee Won-il
Lee Won-il (; born December 12, 1979) is a South Korean chef and television personality.

 He is currently a cast member in the variety show "Please Take Care of My Refrigerator".

 He is owner and chef of "Dear.

 bread".
</item 3>

<item 4>
Graph query row:
singer: Dappy
nationality: English
nationality_place: None
nationality_country: None
birthplace: None
birthplace_country: None
</item 4>

<item 5>
Eunhyuk
Lee Hyuk-jae (born April 4, 1986), better known by his stage name Eunhyuk, is a South Korean singer-songwriter and actor.

 He is a member of the South Korean boy group Super Junior and their subgroups, Super Junior-T and Super Junior-H.

 In 2011, he joined Super Junior's Mandopop subgroup, Super Junior-M and the duo Donghae & Eunhyuk; active in China and Japan respectively.
</item 5>

<item 6>
As One (musical duo)
As One () is a South Korean R&B duo consisting of Korean-American singers Lee Min-young (known as Lee Min) and Chae Da-hee (known as Crystal).

 They are signed to record label Brand New Music.
</item 6>

<item 7>
Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Super Junior-D&E, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 7>

<item 8>
Kim Dong-jun
Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.

 He is a member of the South Korean boyband and its sub-group ZE:A Five and ZE:A J.
</item 8>

<item 9>
List of songs written by Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Donghae & Eunhyuk, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 9>

<item 10>
Graph query row:
singer: Lee Dong-hae
nationality: South Korean
nationality_place: South Korea
nationality_country: None
birthplace: None
birthplace_country: None
</item 10>
=== END ITEM id=5140e5a1e721917f ===

=== ITEM id=8494bf035aeba914 ===
Gold sentences and facts:
Sentence 0: Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer, musician, director, actor, and comedian who holds dual British and American citizenship.
  Fact 0.0: Christopher Haden-Guest is the 5th Baron Haden-Guest.
  Fact 0.1: Christopher Haden-Guest was born on February 5, 1948.
  Fact 0.2: Christopher Haden-Guest is simply known as Christopher Guest.
  Fact 0.3: Christopher Guest is an English-American screenwriter.
  Fact 0.4: Christopher Guest is an English-American composer.
  Fact 0.5: Christopher Guest is an English-American musician.
  Fact 0.6: Christopher Guest is an English-American director.
  Fact 0.7: Christopher Guest is an English-American actor.
  Fact 0.8: Christopher Guest is an English-American comedian.
  Fact 0.9: Christopher Guest holds dual British and American citizenship.
Sentence 1:  Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.0: Christopher Guest is most widely known in Hollywood for having written his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.1: Christopher Guest is most widely known in Hollywood for having directed his series of comedy films shot in mock-documentary (mockumentary) style.
  Fact 1.2: Christopher Guest is most widely known in Hollywood for having starred in his series of comedy films shot in mock-documentary (mockumentary) style.
Sentence 2:  He also directed dark comedies, children's movies, and historical pictures.
  Fact 2.0: Christopher Guest directed dark comedies.
  Fact 2.1: Christopher Guest directed children's movies.
  Fact 2.2: Christopher Guest directed historical pictures.

Retrieved items:
<item 1>
Atom Egoyan
Atom Egoyan, {'1': ", '2': ", '3': ", '4': "} (born July 19, 1960) is a Canadian stage and film director, writer and producer.

 Egoyan made his career breakthrough with "Exotica" (1994), a film set primarily in and around the fictional Exotica strip club.

 Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest
commercial success is the erotic thriller "Chloe" (2009).
</item 1>

<item 2>
The entries highlight acclaimed Israeli cinema: *The Band’s Visit* (2007), directed by Eran Kolirin, and *Hunting Elephants* (2013), Reshef Levi’s crime comedy
starring Sasson Gabai, Moni Moshonov and Patrick Stewart. *Yana’s Friends* (1999) won 10 Ophir Awards and Karlovy Vary’s Crystal Globe. Joseph Cedar won Cannes’
2011 screenplay prize for *Footnote
</item 2>

<item 3>
The entries profile filmmakers, producers, performers, and artists: Tim Bedore is a Chicago-born comedian; Rogue One (2016) depicts rebels stealing Death Star
plans; Frederick K. Keller, Peter Guber, Baltasar Kormákur, Steve Rash, Jonathan Kaplan, Kurt Voss, Jim Sharman, Barry Sandler, and Howard Rosenman are
directors or producers. Other highlights include Titan A.E.’s commercial failure; Vito Russo’s documentary; Scott Free’s founding by the
</item 3>

<item 4>
Lev Kuleshov (1899–1970), Soviet filmmaker and theorist, pioneered Soviet montage, the Kuleshov effect, and creative geography. Hermann Brunn (1862–1939),
German mathematician, contributed to convex geometry and knot theory. Mercury’s 165-km Chaikovskij crater honors composer Pyotr Tchaikovsky; a 1970 Soviet
biopic about him earned two Oscar nominations. Søren Kierkegaard (
</item 4>

<item 5>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 5>

<item 6>
David Jablin
David Jablin is an award winning comedy producer and director and one of the pioneers of original television programming for Pay Television.

 In 1981 he created and produced the innovative comedy anthology series "Likely Stories" for HBO/Cinemax, which featured the directing debuts of Rob Reiner,
Danny DeVito, Christopher Guest, and Billy Crystal.
</item 6>

<item 7>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 7>

<item 8>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 8>

<item 9>
Welcome, or No Trespassing
Welcome, or No Trespassing (Russian: Добро пожаловать, или Посторонним вход воспрещён , "Dobro pozhalovat, ili Postoronnim vkhod vospreshchyon " ) is a Soviet
movie by Elem Klimov made in 1964.

 It is a satirical comedy about the excessive restrictions that children face during their vacation in a Young Pioneer camp, imposed by their masters.

 Most of the actors are children, while the protagonist is the director Dynin, played by Yevgeniy Yevstigneyev.

 The film was selected to be screened in the Cannes Classics section of the 2015 Cannes Film Festival.
</item 9>

<item 10>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 10>
=== END ITEM id=8494bf035aeba914 ===

=== ITEM id=7913b85c1443bab9 ===
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
Distortion Mirrors
Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
</item 1>

<item 2>
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
</item 2>

<item 3>
OK Go (2001 EP)
OK Go or The Pink EP (officially known as OKGoCD.002) is the second EP by American rock band OK Go.
</item 3>

<item 4>
OK Go videography
The success of OK Go's music first won the band the 14th Annual Webby Special Achievement Award for Film and Video Artist of the Year.

 The video for "This Too Shall Pass" was named both "Video of the Year" and "Best Rock Video" at the 3rd annual UK Music Video Awards.

"This Too Shall Pass" won the LA Film Fest's Audience Award for Best Music Video, UK MVA Awards – Music Video of the Year Winner 2010, among others.
</item 4>

<item 5>
OK Go (album)
OK Go is the debut studio album by American rock band OK Go.

 It was released in September 2002.

 The cover was created by designer Stefan Sagmeister.
</item 5>

<item 6>
The Writing's on the Wall (OK Go song)
"The Writing's on the Wall" is a song by American rock band OK Go.

 It was released on June 17, 2014, as part of the band's EP "Upside Out", and is also the first single from the band's fourth studio album "Hungry Ghosts".

 On the same day, the band released a music video in which the members use props to create optical illusions, reflecting the song's description of a
relationship that fails because the couple has different points of view.

 Like previous OK Go videos, it is structured as a one-shot music video.

 The many YouTube views of the video caused the song to debut in the top ten of the US "Billboard" Hot Rock Songs chart, as well as number one on the Bubbling
Under Hot 100 Singles chart.
</item 6>

<item 7>
Twelve Days of OK Go
Twelve Days of OK Go is a compilation album by American rock band OK Go.

 It was released on December 31, 2012.

 OK Go started releasing the songs on December 10, with one song released each weekday.

 The last song, a cover of "Any Time at All", was released on Christmas.

 A bonus track, a cover of "This Will Be Our Year," was released on New Year's Eve.
</item 7>

<item 8>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 8>

<item 9>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 9>

<item 10>
OK Go (2000 EP)
OK Go or The Brown EP (officially known as OKGoCD.001) is the first EP by American rock band OK Go.
</item 10>
=== END ITEM id=7913b85c1443bab9 ===

=== ITEM id=867d4dde3b00d8fd ===
Gold sentences and facts:
Sentence 0: Age of Chance were a British alternative rock-dance crossover band from Leeds, England active from 1985 to 1991.
  Fact 0.0: Age of Chance was a British alternative rock-dance crossover band.
  Fact 0.1: Age of Chance was from Leeds, England.
  Fact 0.2: Age of Chance was active from 1985 to 1991.
Sentence 1: Vector was a rock band formed in Sacramento, California in the early 1980s by Jimmy Abegg, Steve Griffith and Charlie Peacock.
  Fact 1.0: Vector was a rock band.
  Fact 1.1: Vector was formed in Sacramento, California.
  Fact 1.2: Vector was formed in the early 1980s.
  Fact 1.3: Vector was formed by Jimmy Abegg.
  Fact 1.4: Vector was formed by Steve Griffith.
  Fact 1.5: Vector was formed by Charlie Peacock.

Retrieved items:
<item 1>
AES+F
AES+F is an artist collective of four Russian artists: Tatiana Arzamasova (1955), Lev Evzovich (1958), Evgeny Svyatsky (1957), and Vladimir Fridkes (1956).

 It was first formed as AES Group in 1987 by Arzamasova, Evzovich, and Svyatsky, becoming AES+F when Fridkes joined in 1995.

 The collective works in photography, video, installation, and animation, as well as more traditional media, such as painting, drawing, and sculpture.

 AES+F's early work included performance, installation, painting, and illustration.

 Well known for their monumental video-art installations that Gareth Harris describes as "monumental painting set in motion", AES+F create grand visual
narratives that explore contemporary global values, vices and conflicts.
</item 1>

<item 2>
Kinetic Records
Kinetic Records was a New York City-based record label founded by Steve Lau (founding member of Sire Records act the Ocean Blue), that launched the careers of
artists and DJs such as Paul Oakenfold, BT, Ray Munns, Deepsky, Kosheen, Tilt, Timo Maas, Grace, Sandra Collins, Billy Thermal, Binary Finary, and Shpongle.

 Other successful artists on the label included Ash, Sasha and John Digweed, South, Faithless, DJ DAN, LTJ Bukem, Dario G, Luomo, Infected Mushroom, Max Graham,
Dave Ralph Music Instructor, and Hybrid.

 It was also well known for its Tranceport series, of which Paul Oakenfold's first installment is considered to be the breakthrough record for the trance genre
with the highest sales of any mix compilation.

 The release of Sasha's Airdrawndagger and Sasha and Digweed's mix compilation "Communicate" were also big releases for the label.

 Kinetic was the exclusive distributor of the Perfecto label in the US as well as the Another Late Night compilation series, which featured Howie B., Fila
Brazillia, and Rae and Christian.
</item 2>

<item 3>
Multivariate random variable
In probability, and statistics, a multivariate random variable or random vector is a list of mathematical variables each of whose value is unknown, either
because the value has not yet occurred or because there is imperfect knowledge of its value.

 The individual variables in a random vector are grouped together because they are all part of a single mathematical system — often they represent different
properties of an individual statistical unit.

 For example, while a given person has a specific age, height and weight, the representation of these features of "an unspecified person" from within a group
would be a random vector.

 Normally each element of a random vector is a real number.
</item 3>

<item 4>
Vector (band)
Vector was a rock band formed in Sacramento, California in the early 1980s by Jimmy Abegg, Steve Griffith and Charlie Peacock.

 The band had several drummers over the years, including Aaron Smith and Bruce Spencer, both of whom also played drums for The 77s.
</item 4>

<item 5>
Age of Chance
Age of Chance were a British alternative rock-dance crossover band from Leeds, England active from 1985 to 1991.

 They were perhaps most known for their mutant metallic cover of Prince's "Kiss" which topped the UK Indie Chart in 1986, and peaked at No. 50 in the UK Singles
Chart in January the following year.

 Despite signing for major label Virgin, and being favourites with the UK music press, they never enjoyed a major hit in the UK, although "Don't Get Mad… Get
Even" reached No. 8 on the US "Billboard" Hot Dance/Club Play chart.
</item 5>

<item 6>
Graph query row:
band: Age of Chance
bandLabels: ['__KGBuilder__', '__Entity__', 'Organization']
relatedBand: Vector
relatedBandLabels: ['__KGBuilder__', '__Entity__', 'Organization']
directRelationship: None
sharedEntity: None
firstRelationship: None
secondRelationship: None
</item 6>

<item 7>
Word2vec
Word2vec is a group of related models that are used to produce word embeddings.

 These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words.

 Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus
being assigned a corresponding vector in the space.

 Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the
space.
</item 7>

<item 8>
King Electric
King Electric is Chris Isepp and Peter Hartwig.

 They took their first musical attempts at the end of 1980s / beginning 1990s and first met each other at the upcoming and flourishing Viennese techno scene
back in '92.

 While Hartwig was touring with the well known noise-pop band the Basket Boys, Isepp made first attentions as Electro DJ & Producer.
</item 8>

<item 9>
Chao (Sonic)
Chao ( ; Japanese: チャオ ) are fictional life forms in the "Sonic the Hedgehog" video game series published by Sega.

 They are small, childlike creatures that go through a complex life cycle and exist in several visual forms depending on how they are raised.

 Developer Sonic Team incorporated Chao into the games to encourage players to explore levels and support the good–evil dichotomy of "Sonic Adventure 2".
</item 9>

<item 10>
New Valley LLC
New Valley LLC is an investment company owned by Vector Group.

 Western Union was renamed New Valley in the early 1990s as the corporate parent of all Western Union related businesses.

 In 1994, New Valley sold its Western Union businesses to First Financial Management (later acquired by First Data) for $1.2 Billion.

 After more than 100 years of being a publicly traded entity, New Valley became a wholly owned subsidiary of Vector Group on December 13, 2005 when Vector Group
acquired the remaining 42.3% of New Valley's common shares that it did not already own.

 New Valley presently has a 70% stake in Douglas Elliman Realty, the largest residential real estate brokerage in the New York metropolitan area.

 The company also has stakes in various hotels and golf courses in the United States.

 In conjunction with the spin-off of its Ladenburg Thalmann subsidiary New Valley retained a 7.6% stake in Ladenburg Thalmann Financial Services, an investment
bank and stock brokerage based in Miami.
</item 10>
=== END ITEM id=867d4dde3b00d8fd ===

=== ITEM id=8e9d89c7705b715a ===
Gold sentences and facts:
Sentence 0: Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the electronica band Republica.
  Fact 0.0: Samantha Marie Sprackling was born on 3 June 1968.
  Fact 0.1: Samantha Marie Sprackling was born in Lagos, Nigeria.
  Fact 0.2: Samantha Marie Sprackling is more commonly known as Saffron.
  Fact 0.3: Samantha Marie Sprackling is a singer.
  Fact 0.4: Samantha Marie Sprackling is a songwriter.
  Fact 0.5: Samantha Marie Sprackling is an actress.
  Fact 0.6: Samantha Marie Sprackling is the lead singer of the electronica band Republica.
Sentence 1: Myles Richard Kennedy (born Myles Richard Bass; November 27, 1969) is an American musician, singer, and songwriter, best known as the lead vocalist and rhythm guitarist of the rock band Alter Bridge, and as the lead vocalist in guitarist Slash's backing band, known as Myles Kennedy and the Conspirators.
  Fact 1.0: Myles Richard Kennedy was born Myles Richard Bass.
  Fact 1.1: Myles Richard Kennedy was born on November 27, 1969.
  Fact 1.2: Myles Richard Kennedy is an American musician.
  Fact 1.3: Myles Richard Kennedy is an American singer.
  Fact 1.4: Myles Richard Kennedy is an American songwriter.
  Fact 1.5: Myles Richard Kennedy is best known as the lead vocalist and rhythm guitarist of the rock band Alter Bridge.
  Fact 1.6: Myles Richard Kennedy is the lead vocalist in guitarist Slash's backing band.
  Fact 1.7: Slash's backing band is known as Myles Kennedy and the Conspirators.
Sentence 2:  A former guitar instructor from Spokane, Washington, he has worked as a session musician and songwriter, making both studio and live appearances with several artists, and has been involved with several projects throughout his career.
  Fact 2.0: Myles Richard Kennedy is a former guitar instructor.
  Fact 2.1: Myles Richard Kennedy is from Spokane, Washington.
  Fact 2.2: Myles Richard Kennedy has worked as a session musician.
  Fact 2.3: Myles Richard Kennedy has worked as a songwriter.
  Fact 2.4: Myles Richard Kennedy has made studio appearances with several artists.
  Fact 2.5: Myles Richard Kennedy has made live appearances with several artists.
  Fact 2.6: Myles Richard Kennedy has been involved with several projects throughout his career.

Retrieved items:
<item 1>
Myles Kennedy
Myles Richard Kennedy (born Myles Richard Bass; November 27, 1969) is an American musician, singer, and songwriter, best known as the lead vocalist and rhythm
guitarist of the rock band Alter Bridge, and as the lead vocalist in guitarist Slash's backing band, known as Myles Kennedy and the Conspirators.

 A former guitar instructor from Spokane, Washington, he has worked as a session musician and songwriter, making both studio and live appearances with several
artists, and has been involved with several projects throughout his career.
</item 1>

<item 2>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 2>

<item 3>
World on Fire World Tour
The World on Fire World Tour is the third concert tour by lead guitarist Slash as a solo artist, which started in July 2014 and was scheduled to resume in late
2015, in support of Slash's third solo album "World on Fire".

 The tour features the same backing band that performed with Slash during his two last tours, billed as "Slash featuring Myles Kennedy and The Conspirators",
featuring Myles Kennedy handling lead vocal, bassist Todd Kerns, drummer Brent Fitz and rhythm guitarist Frank Sidoris.
</item 3>

<item 4>
Myles Kennedy discography
The full discography of rock musician Myles Kennedy consists of eleven studio albums, two concert films, four live albums, two extended plays, and thirteen
singles in total, in addition to eleven studio tracks that he has appeared on as a featured artist, one of which was a single.

 Born in Boston on November 27, 1969, Kennedy is currently a member of the rock band Alter Bridge, with whom he has released four studio albums, two concert
films, and several singles.

 He is also the frontman of Slash's touring group, and with Slash he has released a live album, "Live in Manchester", the first of a series of live albums
released throughout the summer of 2010, and "", another live album released in 2011.

 In 2012, he released a collaboration studio album with Slash titled "Apocalyptic Love", which is billed to Slash featuring Myles Kennedy and the Conspirators,
as well as the 2014 followup titled "World on Fire".

 With The Mayfield Four, he released two studio albums, two extended plays, and four singles; with Citizen Swing, two studio albums; and with Cosmic Dust, one
studio album.
</item 4>

<item 5>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 5>

<item 6>
List of songs recorded by Myles Kennedy
Myles Kennedy is an American rock musician.

 Born in Boston, Massachusetts, he originally began his musical career in Spokane, Washington as the guitarist in jazz band Cosmic Dust.

 After recording 1991's "Journey", he left to form alternative rock band Citizen Swing in 1992.

 The group released two albums, 1992's "Cure Me with the Groove" and 1996's "Deep Down", on both of which Kennedy was co-credited for songwriting.

 Kennedy's next band was The Mayfield Four, which he co-founded with guitarist Craig Johnson, bassist Marty Meisner and drummer Zia Uddin in 1996.

 The frontman wrote most of the material on the band's 1998 debut "Fallout" and all of the songs on 2001's "Second Skin".
</item 6>

<item 7>
Bent to Fly
"Bent to Fly" is a song by American hard rock guitarist Slash, featuring vocalist Myles Kennedy and backing band The Conspirators.

 Written by Slash and Kennedy, it was released as the second single from the guitarist's third solo album (the second with Kennedy and The Conspirators), "World
on Fire".

 The song was used as the theme song for the 2014 National Rugby League Finals series, and Slash performed the song live at ANZ Stadium as pre-show
entertainment for the league's grand final that year.
</item 7>

<item 8>
World on Fire (Slash song)
"World on Fire" is a song by American hard rock guitarist Slash, featuring vocalist Myles Kennedy and backing band The Conspirators.

 Written by Slash and Kennedy, it is the title track of the guitarist's third solo album (the second with Kennedy and The Conspirators) "World on Fire".

 Released as the album's lead single, "World on Fire" topped the US "Billboard" Mainstream Rock Songs chart.
</item 8>

<item 9>
Graph query row:
entity: Myles Kennedy
occupation: ['rock musician', 'guitarist', 'frontman']
relationship: BORN_IN
role: None
related: Boston
</item 9>

<item 10>
Apocalyptic Love World Tour
The Apocalyptic Love World Tour is the second concert tour by ex-Guns N' Roses lead guitarist Slash as a solo artist.

 The tour features the same backing band that toured with Slash during his first solo tour, though this time the act is billed as "Slash featuring Myles Kennedy
and The Conspirators".

 The tour is in support of Slash's second solo album "Apocalyptic Love", which features Myles Kennedy handling lead vocal duties on all songs, as well as his
touring band members: bassist Todd Kerns, drummer Brent Fitz and rhythm guitarist Frank Sidoris.
</item 10>
=== END ITEM id=8e9d89c7705b715a ===

=== ITEM id=1fa240898be4b632 ===
Gold sentences and facts:
Sentence 0: August or Auguste Brancart (21 July 1851 - 1894?)
  Fact 0.0: August Brancart is also known as Auguste Brancart.
  Fact 0.1: August Brancart was born on 21 July 1851.
  Fact 0.2: The end date given for August Brancart is 1894, with a question mark.
Sentence 1:  He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
  Fact 1.0: Brancart published work of the Decadent movement.
  Fact 1.1: Brancart published "Monsieur Vénus" by Rachilde.
  Fact 1.2: "Monsieur Vénus" by Rachilde is an example of Decadent movement work.
Sentence 2: The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and artificiality.
  Fact 2.0: The Decadent Movement was an artistic and literary movement.
  Fact 2.1: The Decadent Movement was a late 19th-century movement.
  Fact 2.2: The Decadent Movement was centered in Western Europe.
  Fact 2.3: The Decadent Movement followed an aesthetic ideology of excess.
  Fact 2.4: The Decadent Movement followed an aesthetic ideology of artificiality.

Retrieved items:
<item 1>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 1>

<item 2>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 2>

<item 3>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 3>

<item 4>
Decadent movement
The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and
artificiality.

 The visual artist Félicien Rops's body of work and Joris-Karl Huysmans's novel "Against Nature" (1884) are considered the prime examples of the decadent
movement.

 It first flourished in France and then spread throughout Europe and to the United States.

 The movement was characterized by self-disgust, sickness at the world, general skepticism, delight in perversion and employment of crude humor and a belief in
the superiority of human creativity over logic and the natural world.
</item 4>

<item 5>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 5>

<item 6>
Karl Wilhelm Friedrich Schlegel
Karl Wilhelm Friedrich (after 1814: von) Schlegel (10 March 1772 – 12 January 1829), usually cited as Friedrich Schlegel, was a German poet, literary critic,
philosopher, philologist and Indologist.

 With his older brother, August Wilhelm Schlegel, he was one of the main figures of the Jena romantics.

 He was a zealous promoter of the Romantic movement and inspired Samuel Taylor Coleridge, Adam Mickiewicz and Kazimierz Brodziński.

 Schlegel was a pioneer in Indo-European studies, comparative linguistics, morphological typology, and was the first to notice what became known as Grimm's law.
</item 6>

<item 7>
Marcel J. Melançon
Marcel J. Mélançon is a Canadian philosopher and scientist.

 He was born in Saint-Barnabé, Quebec, in 1938, and is mostly known for his book on the French philosopher Albert Camus "Albert Camus, An Analysis of his
thought" (Albert Camus.

 Analyse de sa pensée).

 He has mostly worked together with the biologist Richard D. Lambert.
</item 7>

<item 8>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 8>

<item 9>
Hermann Brunn
Karl Hermann Brunn (1 August 1862 – 20 September 1939) was a German mathematician, known for his work in convex geometry (see Brunn–Minkowski inequality) and in
knot theory.

 Brunnian links are named after him, as his 1892 article "Über Verkettung" included examples of such links.
</item 9>

<item 10>
The Exception and the Rule
The Exception and the Rule (in German Die Ausnahme und die Regel) is a short play by German playwright Bertolt Brecht and is one of several "Lehrstücke"
(Teaching plays) he wrote around 1929/30.

 The objective of Brecht's Lehrstücke was that they be taken on tour and performed in schools or in factories to educate the masses about socialist politics.
</item 10>
=== END ITEM id=1fa240898be4b632 ===

=== ITEM id=1bcf28008847ab6c ===
Gold sentences and facts:
Sentence 0:  Jan Kodeš and Margaret Court won the singles titles.
  Fact 0.0: Jan Kodeš won the singles titles.
  Fact 0.1: Margaret Court won the singles titles.
Sentence 1:  She is currently a Christian minister in Perth, Western Australia.
  Fact 1.0: A woman referred to in this sentence is currently a Christian minister.
  Fact 1.1: The Christian minister referred to in this sentence is based in Perth, Western Australia.

Retrieved items:
<item 1>
Jorge Lozano
In Davis Cup play, represented Mexico for 15 years, 1981–1995, won 12 doubles matches, and lost 12 as well.

 He lost 11 singles matches, and won 8, making his overall win / loss record at the Davis Cup 20–23.

 Lozano won his first title at Forest Hills in 1988, with his partner Todd Witsken, and won his last at Athens in 1993.

 He resides in Guadalajara, Mexico where he directs a tennis academy for kids and also coaches the men and women tennis team at the University Tec de Monterrey.

 He is currently the captain of the Mexican Davis Cup Team.
</item 1>

<item 2>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 2>

<item 3>
1982 French Open – Men's Doubles
The Men's Doubles tournament at the 1982 French Open was held from 24 May until 6 June 1982 on the outdoor clay courts at the Stade Roland Garros in Paris,
France.

 Sherwood Stewart and Ferdi Taygan won the title, defeating Hans Gildemeister and Belus Prajoux in the final.
</item 3>

<item 4>
1972 French Open – Men's Singles
Jan Kodes was the two-time defending champion, but lost in the quarterfinals to Patrick Proisy.
</item 4>

<item 5>
1970 French Open
The 1970 French Open was a tennis tournament that was held at the Stade Roland Garros in Paris in France from 25 May through 7 June 1970.

 It was the 69th edition of the French Open, the 40th to be open to foreign competitors, and the second Grand Slam of the year.

 Jan Kodeš and Margaret Court won the singles titles.
</item 5>

<item 6>
Martin Legner
Martin Legner (born December 17, 1961 in Tyrol) is a professional Austrian wheelchair tennis player who has been ranked number one for wheelchair doubles.

 Legner has won the Australian Open doubles title four times with Robin Ammerlaan since 2000, and has won eight doubles titles.

 Six of them are Australian Open Titles, the other two being French Open titles.

 He has won one French Open and one Australian Open singles title.

 He has represented his country at every Summer Paralympics since 1992, and has competed in both singles and doubles at all of those Games.

 His favorite surfaces are clay and hard court.

 As of April 7, 2007 his highest singles rank was number three.

 Also of that date, he was ranked number six for singles and doubles.

 His wheelchair is manufactured by Kronbickler.

 Legner is a fan of Bob Marley, and listens to his music aside from tennis.
</item 6>

<item 7>
Margaret Court
Margaret Court {'1': ", '2': ", '3': ", '4': "} (née Smith; born 16 July 1942), also known as Margaret Smith Court, is a retired Australian tennis player and
former world No. 1.

 She is currently a Christian minister in Perth, Western Australia.

 In tennis, she amassed more major titles than any other player in history.
</item 7>

<item 8>
Carlos Fernandes (tennis)
At Roland Garros 1962, Fernandes lost to Lundquist in round two.

 At Wimbledon he lost in round one to Neale Fraser's brother John.

 At 1963 French Open he lost in round one to Christian Duxin.

 In round one of Wimbledon he led a young Arthur Ashe by two sets to 0, but lost in five sets.

 At Wimbledon 1964 he lost in round two to Gene Scott.

 An ankle injury in 1964 took its' toll and Fernandes played less after that.

 As Open tennis arrived, Fernandes was reaching the end of his career.

 He lost in round one of 1968 French Open to Bernard Montrenaud.

 At Wimbledon he lost in round one to Mike Sangster.
</item 8>

<item 9>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 9>

<item 10>
Jorge Lozano
Jorge Lozano (born 17 May 1963 in San Luis Potosí) is a retired professional tennis player from Mexico.

 He reached his highest doubles ranking of World No. 4 in August 1988.

 His highest singles ranking was World No. 51, achieved the following month.

 During his career, he won two mixed doubles titles at the French Open: in 1990 with Arantxa Sánchez, and in 1988 with Lori McNeil.

 He reached the round of 16 in singles at the 1988 US Open and also the semifinal in doubles that same year.

 Qualified twice for the Doubles Masters at the Royal Albert Hall in London and reached the semifinals in 1988.

 In his career, he won nine doubles titles, but no singles titles.

 He turned professional in 1986, and in his career, he won $U.S.739,424 in prize money.

 He was the first player to be beaten by Pete Sampras in the main draw of a Grand Slam tournament, at the French Open in 1989.
</item 10>
=== END ITEM id=1bcf28008847ab6c ===

=== ITEM id=31b8bfe4b2eff432 ===
Gold sentences and facts:
Sentence 0: Ranger Rick, originally Ranger Rick's Nature Magazine, is a children’s nature magazine that is published by the United States National Wildlife Federation.
  Fact 0.0: Ranger Rick was originally called Ranger Rick's Nature Magazine
  Fact 0.1: Ranger Rick is a children's nature magazine
  Fact 0.2: Ranger Rick is published by the United States National Wildlife Federation
Sentence 1: TENNIS is an American sports magazine owned by the Sinclair Broadcast Group.
  Fact 1.0: TENNIS is an American sports magazine
  Fact 1.1: TENNIS is owned by the Sinclair Broadcast Group

Retrieved items:
<item 1>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 1>

<item 2>
National Review
National Review (NR) is an American semi-monthly conservative editorial magazine focusing on news and commentary pieces on political, social, and cultural
affairs.

 The magazine was founded by the author William F. Buckley Jr. in 1955.

 It is currently edited by Rich Lowry.
</item 2>

<item 3>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 3>

<item 4>
El Gráfico
El Gráfico is a classic monthly Argentine sports magazine.

 The magazine was originally published as a weekly newspaper in May 1919, and then turned to a sports magazine exclusively.

 It has been scheduled monthly from 2002.

 "El Gráfico" is one of the most regarded sports magazines not only in Argentina but in Latin America.

 The magazine was nicknamed "La Biblia del deporte" ("The Bible of sports") due to its chronicles, notable journalists and collaborators and its photography.
</item 4>

<item 5>
Graph query row:
publication: Ranger Rick
kind: children's magazine
published_on: None
genre: None
language: None
relationship: PART_OF
related: National Wildlife Federation
</item 5>

<item 6>
Lorin Thompson
Lorin Thompson, Lorin Hartwell Jr Thompson, (1911–1997), was a muralist, artist, and creator and illustrator of the character "Ranger Rick" for the National
Wildlife Federation's children's magazine, Ranger Rick.
</item 6>

<item 7>
Ranger Rick
Ranger Rick, originally Ranger Rick's Nature Magazine, is a children’s nature magazine that is published by the United States National Wildlife Federation.

 The magazine offers feature articles and activities for children, ages 7 and up, in order to spark their interest in the outdoors and become more actively
involved in the environment.

 The magazine's primary intention is to instill a passion for nature and promote activity outdoors.

 NWF also publishes two companion magazines, "Ranger Rick Jr.", which is aimed at ages 4–7, and "Ranger Rick Cub", which is aimed at kids 0–4 years old.)
</item 7>

<item 8>
Tennis (magazine)
TENNIS is an American sports magazine owned by the Sinclair Broadcast Group.

 There are eight issues per year covering the world of tennis.

 The "tennis.com" website is an offshoot of the magazine and part of the same media company.
</item 8>

<item 9>
Memphis Sport
Memphis Sport is a sports and fitness magazine featuring articles on local and regional teams, players and events.

 In addition, "Memphis Sport" focuses on health and fitness for an active lifestyle.

 The magazine debuted in Memphis, Tennessee in July 2006.

 Local franchises covered include the University of Memphis Tigers, Memphis Redbirds, Mississippi Riverkings and Tennessee Titans.
</item 9>

<item 10>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 10>
=== END ITEM id=31b8bfe4b2eff432 ===

=== ITEM id=ad6edd8e6db71e10 ===
Gold sentences and facts:
Sentence 0:  The album features a pair of duets with Waylon Jennings, one of which was the "There Ain't No Good Chain Gang" single; it was one of Cash's first collaborations with Jennings, and the two recorded songs together throughout the 1980s, including a separate album entitled "Heroes".
  Fact 0.0: The album features a pair of duets with Waylon Jennings.
  Fact 0.1: One of the duets with Waylon Jennings was the "There Ain't No Good Chain Gang" single.
  Fact 0.2: The duet "There Ain't No Good Chain Gang" was one of Johnny Cash's first collaborations with Waylon Jennings.
  Fact 0.3: Johnny Cash and Waylon Jennings recorded songs together throughout the 1980s.
  Fact 0.4: Johnny Cash and Waylon Jennings recorded a separate album entitled "Heroes".
Sentence 1:  Cash and Jennings would also work together as The Highwaymen with Willie Nelson and Kris Kristofferson.
  Fact 1.0: Johnny Cash and Waylon Jennings would also work together as The Highwaymen.
  Fact 1.1: The Highwaymen included Willie Nelson.
  Fact 1.2: The Highwaymen included Kris Kristofferson.
Sentence 2: The Highwaymen was a country music supergroup composed of four of the genre's biggest artists well known for their pioneering influence on the outlaw country subgenre: Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.
  Fact 2.0: The Highwaymen was a country music supergroup.
  Fact 2.1: The Highwaymen was composed of four of the country music genre's biggest artists.
  Fact 2.2: The four artists were well known for their pioneering influence on the outlaw country subgenre.
  Fact 2.3: The Highwaymen members were Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.
Sentence 3:  Active as a group between 1985 and 1995, these four artists recorded three major label albums as The Highwaymen: two on Columbia Records and one for Liberty Records.
  Fact 3.0: The Highwaymen was active as a group between 1985 and 1995.
  Fact 3.1: The Highwaymen recorded three major label albums.
  Fact 3.2: Two of The Highwaymen's major label albums were on Columbia Records.
  Fact 3.3: One of The Highwaymen's major label albums was for Liberty Records.

Retrieved items:
<item 1>
Back in the High Life Again
"Back in the High Life Again" is a 1986 single written by Steve Winwood and Will Jennings and performed by Winwood.

 The single was included on his album "Back in the High Life" and included backing vocals by James Taylor as well as a prominent mandolin played by Winwood.

 "Back in the High Life Again" was Winwood's second number one on the Adult Contemporary chart.

 The single stayed at number one for three weeks and went to number 13 on the "Billboard" Hot 100.
</item 1>

<item 2>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 2>

<item 3>
The Closer You Get...
The Closer You Get... is the seventh studio album by country music band Alabama, released in 1983.

 All three singles from this album — "The Closer You Get", "Lady Down on Love" and "Dixieland Delight" — reached Number One on the Hot Country Songs charts in
1983.

 "She Put the Sad in All His Songs" was also recorded by Ronnie Dunn (who formed the duo Brooks & Dunn with Kix Brooks in 1991) and was released by him as a
single in 1983.
</item 3>

<item 4>
Burn Your Playhouse Down – The Unreleased Duets
Burn Your Playhouse Down – The Unreleased Duets is the 60th and final studio album by American country music singer George Jones released on August 19, 2008 on
the Bandit Records label.

 It features duets never before released, including some that were cut from his 1994 duets album "The Bradley Barn Sessions".

 The only new recording in the collection is "You And Me And Time", a song Jones recorded with his daughter by Tammy Wynette, Georgette.

 A music video accompanied the song.

 The album features several duets with artist from outside the country music pantheon, including Mark Knopfler, Leon Russell, and Keith Richards.

 The album peaked at number 15 on the Billboard country albums chart.

 Of the album's title track, Andrew Meuller of "Uncut" opined in July 2013, "The segue from Richards trying to sing like Jones to Jones actually singing like
Jones is hilarious."
</item 4>

<item 5>
Graph query row:
person: Waylon Jennings
relationship: MEMBER_OF
role: None
membership_period: None
status: None
organization: The Highwaymen
founded_on: None
</item 5>

<item 6>
I Would Like to See You Again
I Would Like to See You Again is the 57th album by American country singer Johnny Cash, released on Columbia Records in 1978.

 The title track peaked at #12 on the singles chart, while "There Ain't No Good Chain Gang" reached #2; the album itself peaked at #23.

 The album features a pair of duets with Waylon Jennings, one of which was the "There Ain't No Good Chain Gang" single; it was one of Cash's first
collaborations with Jennings, and the two recorded songs together throughout the 1980s, including a separate album entitled "Heroes".

 Cash and Jennings would also work together as The Highwaymen with Willie Nelson and Kris Kristofferson.
</item 6>

<item 7>
The Highwaymen (country supergroup)
The Highwaymen was a country music supergroup composed of four of the genre's biggest artists well known for their pioneering influence on the outlaw country
subgenre: Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.

 Active as a group between 1985 and 1995, these four artists recorded three major label albums as The Highwaymen: two on Columbia Records and one for Liberty
Records.

 Their Columbia works produced three chart singles, including the number one "Highwayman" in 1985.
</item 7>

<item 8>
There Ain't No Good Chain Gang
"There Ain't No Good Chain Gang" is a song written by Hal Bynum and Dave Kirby, and recorded by American country music artists Johnny Cash and Waylon Jennings.

 It was released in May 1978 as the second single from the album "I Would Like to See You Again".

 The song reached #2 on the "Billboard" Hot Country Singles & Tracks chart.
</item 8>

<item 9>
Graph query row:
person: Johnny Cash
relationship: MEMBER_OF
role: None
membership_period: None
status: None
organization: The Highwaymen
founded_on: None
</item 9>

<item 10>
Leave Them Boys Alone
"Leave Them Boys Alone" is a song recorded by American country music artist Hank Williams Jr. with Waylon Jennings and Ernest Tubb.

 It was released in May 1983 as the second single from Williams' album "Strong Stuff".

 The song reached number 6 on the "Billboard" Hot Country Singles chart.

 It was written by Williams, Dean Dillon, Gary Stewart and Tanya Tucker.

 The song is notable for its combination of two singers associated with the outlaw movement with a country legend from the honky tonk days and golden age of the
Grand Ole Opry.

 Outlaw singers like Williams and Jennings saw themselves as taking country music back to its raw, honky tonk roots, and recording an up tempo song with Tubb
(who would never have received radio airplay in the late 1970s and early 80's) and reaching #6 was a slap in the face to the proponents of the country pop
sound.

 The lyrics of the song, much like Williams' "Family Tradition" echo the sentiment that the outlaw singers and their current escapades were predated by the hard
living honky tonkers of the 1950s such as Hank Williams, Sr.
</item 10>
=== END ITEM id=ad6edd8e6db71e10 ===
