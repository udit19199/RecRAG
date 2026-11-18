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

=== ITEM id=51d2608edb9c295b ===
Gold sentences and facts:
Sentence 0: The 1892 Auburn Tigers football team represented Auburn University in the 1892 college football season.
  Fact 0.0: The 1892 Auburn Tigers football team represented Auburn University.
  Fact 0.1: The 1892 Auburn Tigers football team played in the 1892 college football season.
Sentence 1:  It was the first college football team fielded by the Agricultural and Mechanical College of Alabama, now known as Auburn University.
  Fact 1.0: The 1892 Auburn Tigers football team was the first college football team fielded by the Agricultural and Mechanical College of Alabama.
  Fact 1.1: The Agricultural and Mechanical College of Alabama is now known as Auburn University.
Sentence 2:  With more than 22,000 undergraduate students and a total enrollment of more than 28,000 with 1,260 faculty members, Auburn is the second largest university in Alabama.
  Fact 2.0: Auburn University has more than 22,000 undergraduate students.
  Fact 2.1: Auburn University has a total enrollment of more than 28,000.
  Fact 2.2: Auburn University has 1,260 faculty members.
  Fact 2.3: Auburn University is the second largest university in Alabama.

Retrieved items:
<item 1>
1983 Alabama Crimson Tide football team
The 1983 Alabama Crimson Tide football team (variously "Alabama", "UA", "Bama" or "The Tide") represented the University of Alabama in the 1983 NCAA Division
I-A football season.

 It was the Crimson Tide's 91st overall and 50th season as a member of the Southeastern Conference (SEC).

 The team was led by head coach Ray Perkins, in his first year, and played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in
Birmingham, Alabama.

 They finished the season with a record of eight wins and four losses (8–4 overall, 4–2 in the SEC).

 Ray Perkins, who played as a wide receiver for Bear Bryant in the 1960s, was named as the new head coach at Alabama on December 14, 1982, to succeed Bryant
after his 25-year tenure as Alabama's head coach.
</item 1>

<item 2>
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
</item 2>

<item 3>
1997 Alabama Crimson Tide football team
The 1997 Alabama Crimson Tide football team represented the University of Alabama in the college football season of 1997–98.

 The team's head coach was Mike DuBose, who was entering his first year at Alabama.

 They played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in Birmingham, Alabama and competed in the West Division of the
Southeastern Conference.

 Alabama would finish with a record of 4–7 record in suffering the program's first losing season since the 1984 season.

 The loss against Kentucky marked Alabama's first ever overtime game, as overtime rules for college football had gone into effect the previous season.
</item 3>

<item 4>
Auburn University
Auburn University (AU or Auburn) is a public research university in Auburn, Alabama, United States.

 With more than 22,000 undergraduate students and a total enrollment of more than 28,000 with 1,260 faculty members, Auburn is the second largest university in
Alabama.

 Auburn University is one of the state's two public flagship universities.
</item 4>

<item 5>
2012 Texas A&amp;M vs. Alabama football game
The 2012 Texas A&M vs. Alabama football game was a college football game between the Texas A&M Aggies and Alabama Crimson Tide at Bryant–Denny Stadium in
Tuscaloosa, Alabama.

 This was the game where Texas A&M freshman quarterback and eventual Heisman Trophy winner Johnny Manziel completed 24 of 31 passes, had 253 passing yards, and
two passing touchdowns with 92 rushing yards to help No. 15 Texas A&M upset No. 1 Alabama 29–24, which led him to being the first freshman to win the Heisman
Trophy.
</item 5>

<item 6>
1906 Alabama Crimson White football team
The 1906 Alabama Crimson White football team (variously "Alabama", "UA" or "Bama") represented the University of Alabama in the 1906 Southern Intercollegiate
Athletic Association football season.

 It was Alabama's 14th overall and 11th season as a member of the Southern Intercollegiate Athletic Association (SIAA).

 The team was led by head coach J. W. H. Pollard, in his first year, and played their home games at both the University of Alabama Quad in Tuscaloosa and the
Birmingham Fairgrounds in Birmingham, Alabama.

 They finished the season with a record of five wins and one loss (5–1 overall, 3–1 in the SIAA).
</item 6>

<item 7>
1902 Connecticut Aggies football team
The 1902 Connecticut Aggies football team represented Connecticut Agricultural College, now the University of Connecticut, in the 1902 college football season.

 This was the seventh year that the school fielded a football team.

 The Aggies were led by first year head coach Edwin O. Smith, and completed the season with a record of 4–3.
</item 7>

<item 8>
West Alabama Tigers football
The West Alabama Tigers football program is the intercollegiate American football team for the University of West Alabama located in the U.S. state of Alabama.

 The team competes in the NCAA Division II and are members of the Gulf South Conference.

 West Alabama's first football team was fielded in 1938.

 The team plays its home games at the 7,000 seat Tiger Stadium in Livingston, Alabama.

 The Tigers are coached by Brett Gilliland.
</item 8>

<item 9>
1905 Alabama Crimson White football team
The 1905 Alabama Crimson White football team (variously "Alabama", "UA" or "Bama") represented the University of Alabama in the 1905 Southern Intercollegiate
Athletic Association football season.

 It was Alabama's 13th overall and 10th season as a member of the Southern Intercollegiate Athletic Association (SIAA).

 The team was led by head coach Jack Leavenworth, in his first year, and played their home games at both the University of Alabama Quad in Tuscaloosa and the
Birmingham Fairgrounds in Birmingham, Alabama.

 They finished the season with a record of six wins and four losses (6–4 overall, 4–4 in the SIAA).
</item 9>

<item 10>
1892 Auburn Tigers football team
The 1892 Auburn Tigers football team represented Auburn University in the 1892 college football season.

 It was the first college football team fielded by the Agricultural and Mechanical College of Alabama, now known as Auburn University.

 The squad was first coached by George Petrie.

 Auburn shut out Georgia Tech, 26–0, just two days after being shut out themselves by North Carolina, 64–0.

 The team finished the season with a record of 2–2.
</item 10>
=== END ITEM id=51d2608edb9c295b ===

=== ITEM id=df6f88339f7c8db6 ===
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
Battle of Germantown
The Battle of Germantown was a major engagement in the Philadelphia campaign of the American Revolutionary War.

 It was fought on October 4, 1777, at Germantown, Pennsylvania, between the British Army led by Sir William Howe, and the American Continental Army, with the
2nd Canadian Regiment, under George Washington.
</item 1>

<item 2>
Province of New York
The Province of New York (1664–1776) was a British proprietary colony and later royal colony on the northeast coast of North America.

 As one of the Thirteen Colonies, New York achieved independence and worked with the others to found the United States.
</item 2>

<item 3>
New York – New Jersey Line War
The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between
1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
</item 3>

<item 4>
Proprietary colony
A proprietary colony was a type of British colony mostly in North America and the Caribbean in the 17th century.

 In the British Empire, all land belonged to the ruler, and it was his prerogative to divide.

 Therefore, all colonial properties were partitioned by royal charter into one of four types: proprietary, royal, joint stock, or covenant.

 King Charles II used the proprietary solution to reward allies and focus his own attention on Britain itself.

 He offered his friends colonial charters which facilitated private investment and colonial self-government.

 The charters made the proprietor the effective ruler, albeit one ultimately responsible to English law and the king.

 Charles II gave New Netherland to his younger brother The Duke of York, who named it New York.

 He gave an area to William Penn who named it Pennsylvania.
</item 4>

<item 5>
Province of Pennsylvania
The Province of Pennsylvania, also known as the Pennsylvania Colony, was founded in English North America by William Penn on March 4, 1681 as dictated in a
royal charter granted by King Charles II.

 The name Pennsylvania, which translates roughly as "Penn's Woods", was created by combining the Penn surname (in honor of William's father, Admiral Sir William
Penn) with the Latin word "sylvania", meaning "forest land."

 The Province of Pennsylvania was one of the two major Restoration colonies, the other being the Province of Carolina.

 The proprietary colony's charter remained in the hands of the Penn family until the American Revolution, when the Commonwealth of Pennsylvania was created and
became one of the original thirteen states.
</item 5>

<item 6>
Charter colony
Charter colony is one of three classes of colonial government established in the 17th century English colonies in North America, the other classes being
proprietary colony and royal colony.

 The colonies of Rhode Island, Connecticut, and Massachusetts Bay were charter colonies.

 In a charter colony, Britain granted a charter to the colonial government establishing the rules under which the colony was to be governed.

 The charters of Rhode Island and Connecticut granted the colonists significantly more political liberty than other colonies.

 Rhode Island and Connecticut continued to use their colonial charters as their State constitutions after the American Revolution.
</item 6>

<item 7>
Stamp Act Congress
The Stamp Act Congress or First Congress of the American Colonies was a meeting held between October 7 and 25, 1765 in New York City, consisting of
representatives from some of the British colonies in North America; it was the first gathering of elected representatives from several of the American colonies
to devise a unified protest against new British taxation.

 Parliament had passed the Stamp Act, which required the use of specially stamped paper for legal documents, playing cards, calendars, newspapers and dice for
virtually all business in the colonies, and was going into effect on November 1.
</item 7>

<item 8>
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
</item 8>

<item 9>
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
</item 9>

<item 10>
Province of New Jersey
The Province of New Jersey was one of the Middle Colonies of Colonial America and became the U.S. state of New Jersey in 1776.

 The province had originally been settled by Europeans as part of New Netherland, but came under English rule after the surrender of Fort Amsterdam in 1664,
becoming a proprietary colony.

 The English then renamed the province after the Isle of Jersey in the English Channel.

 The Dutch Republic reasserted control for a brief period in 1673–1674.

 After that it consisted of two political divisions, East Jersey and West Jersey, until they were united as a royal colony in 1702.

 The original boundaries of the province were slightly larger than the current state, extending into a part of the present state of New York, until the border
was finalized in 1773.
</item 10>
=== END ITEM id=df6f88339f7c8db6 ===

=== ITEM id=0e32aecdac3d3898 ===
Gold sentences and facts:
Sentence 0:  He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
  Fact 0.0: The person referred to as 'He' has had a cameo acting role as a tailor in The Wolf of Wall Street.
  Fact 0.1: The person referred to as 'He' has had a cameo acting role as a tailor in The Good Shepherd.
Sentence 1: The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir of the same name by Jordan Belfort.
  Fact 1.0: The Wolf of Wall Street is a 2013 film.
  Fact 1.1: The Wolf of Wall Street is American.
  Fact 1.2: The Wolf of Wall Street is a biographical black comedy crime film.
  Fact 1.3: The Wolf of Wall Street was directed by Martin Scorsese.
  Fact 1.4: The Wolf of Wall Street was written by Terence Winter.
  Fact 1.5: The Wolf of Wall Street is based on the memoir of the same name by Jordan Belfort.

Retrieved items:
<item 1>
Leonard Logsdail
Leonard Logsdail (born September 11, 1950 in London, England) is a bespoke tailor, based in Manhattan, New York, specializing in men's suits.

 He is described as one of the finest bespoke tailors in the men’s suit business.

 All of Logsdail's suits are crafted and perfected on-site in his New York City location.

 He is credited for making Savile Row tailoring a local option in New York.

 Logsdail is known for creating high-end suits, including lining jackets with Hermes silk scarves.

 CNBC talk show host, economist and fashion icon Larry Kudlow has his suits made by Leonard Logsdail.

 Logsdail has created suits for award-winning films, and is recognized as one of cinema's most sought-after tailors.

 Logsdail has collaborated with esteemed film directors Steven Spielberg, Robert de Niro, Oliver Stone, Ridley Scott, and Martin Scorcese.

 He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
</item 1>

<item 2>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 2>

<item 3>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 3>

<item 4>
The Life of Emile Zola
The Life of Emile Zola is a 1937 American biographical film about French author Émile Zola, played by Paul Muni and directed by William Dieterle.

 It has the distinction of being the second biographical film to win the Oscar for Best Picture.

 It premiered at the Los Angeles Carthay Circle Theatre to great success both critically and financially.

 Contemporary reviews cited it the best biographical film made up to that time.

 In 2000, it was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or
aesthetically significant".
</item 4>

<item 5>
A Walk in the Woods (film)
A Walk in the Woods is a 2015 American biographical comedy-drama film directed by Ken Kwapis, based on the 1998 book/memoir of the same name by Bill Bryson.

 The film stars Robert Redford, Nick Nolte and Emma Thompson.

 The film was released on September 2, 2015, by Broad Green Pictures.
</item 5>

<item 6>
The Wolf of Wall Street (2013 film)
The film marks the director's fifth collaboration with DiCaprio, after "Gangs of New York" (2002), "The Aviator" (2004), "The Departed" (2006) and "Shutter
Island" (2010), as well as his second collaboration with Winter after the television series "Boardwalk Empire" (2010–14).
</item 6>

<item 7>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 7>

<item 8>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 8>

<item 9>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 9>

<item 10>
The Wolf of Wall Street (2013 film)
The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir
of the same name by Jordan Belfort.

 It recounts Belfort's perspective on his career as a stockbroker in New York City and how his firm Stratton Oakmont engaged in rampant corruption and fraud on
Wall Street that ultimately led to his downfall.

 Leonardo DiCaprio (who was also a producer) stars as Belfort, with Jonah Hill as his business partner and friend Donnie Azoff, Margot Robbie as his wife Naomi
Lapaglia and Kyle Chandler as Patrick Denham, the FBI agent who tries to bring him down.

 Matthew McConaughey, Rob Reiner, Jon Favreau, Joanna Lumley and Jean Dujardin also star.
</item 10>
=== END ITEM id=0e32aecdac3d3898 ===

=== ITEM id=2b9e102cefeb1356 ===
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
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 2>

<item 3>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
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
</item 6>

<item 7>
Alf Clausen
Alf Heiberg Clausen (born March 28, 1941) is an American film and television composer.

 He is best known for his work scoring many episodes of "The Simpsons", of which he had been the sole composer between 1990 and 2017.

 Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's
Day Off".
</item 7>

<item 8>
Angelo Badalamenti
Angelo Badalamenti (born March 22, 1937) is an American composer, best known for his work scoring films for director David Lynch, notably "Blue Velvet", the
"Twin Peaks" saga (1990–1992, 2017), "The Straight Story" and "Mulholland Drive".

 Badalamenti received the 1990 Grammy Award for Best Pop Instrumental Performance for his ""Twin Peaks" Theme", and has received a Lifetime Achievement Award
from the World Soundtrack Awards and the Henry Mancini Award from the American Society of Composers, Authors and Publishers.
</item 8>

<item 9>
List of Beelzebub episodes
The television series uses six pieces of theme music, three openings and three endings.

 The first opening theme used between episodes 1 to 10 is "DaDaDa" (だだだ ) by Group Tamashii, while the second opening theme, used from episode 11-23, is "The
First Goodbye" (始まるのは, サヨナラ , Hajimaru no wa, Sayonara ) by On/Off.

 The third opening theme, "Hey!!!"

 by FLOW is used from episodes 24 onward.

 The fourth opening theme Baby U!

 by MBLAQ is used from episodes 36 to 48.

 The fifth opening theme is Only you -Kimi to no Kizuna- by Lc5 is used from episodes 49 onward.

 The first ending theme used for episodes 1-10 is "Answer" by no3b, while the second theme, used from episode 11-23 is "Show of Courage" (つよがり , Tsuyogari ) by
Shoko Nakagawa.
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
=== END ITEM id=2b9e102cefeb1356 ===

=== ITEM id=5bdcc5432bd9bb0d ===
Gold sentences and facts:
Sentence 0:  The documentary explores the history of the mile and a half long stretch of road through West Hollywood known as Sunset Strip.
  Fact 0.0: The documentary explores the history of Sunset Strip.
  Fact 0.1: Sunset Strip is a mile and a half long stretch of road.
  Fact 0.2: Sunset Strip runs through West Hollywood.
Sentence 1:  The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.
  Fact 1.0: The film features some early pioneers of the hardcore punk music scene.
  Fact 1.1: Bad Brains is an early hardcore punk pioneer featured in the film.
  Fact 1.2: Black Flag is an early hardcore punk pioneer featured in the film.
  Fact 1.3: D.O.A. is an early hardcore punk pioneer featured in the film.
  Fact 1.4: Minor Threat is an early hardcore punk pioneer featured in the film.
  Fact 1.5: Minutemen is an early hardcore punk pioneer featured in the film.
  Fact 1.6: SSD is an early hardcore punk pioneer featured in the film.
  Fact 1.7: The film features other early hardcore punk pioneers besides those named.

Retrieved items:
<item 1>
The entries span film, television, documentaries, and film distribution: Philippine series Sine Novela adapts Viva classics; The Doom Generation (1995) is Gregg
Araki’s violent “Teenage Apocalypse” film; Naked Gun 33⅓ concludes its comedy trilogy, while Naked City (1958–63) is a semi-documentary police drama. Night Life
is a zombie comedy; The Runaways portrays the band’s formation and Currie-Jett relationship. Consolidated Film Industries was
</item 1>

<item 2>
The entries cover documentaries on Boston hardcore (All Ages, 2012), punk (Punk’s Not Dead, 2007; American Hardcore, 2006), bipolar disorder, artistic aging,
slavery, archaeology, bookstores, and Charles Bukowski; plus Barbara Hammer’s autobiographical Tender Fictions. Fiction films include Bakshi’s acclaimed X-rated
Heavy Traffic, Heavyweights, several American Pie and Bring It On sequels, Schrader’s Hardcore, Blue Collar, and The
</item 2>

<item 3>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 3>

<item 4>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 4>

<item 5>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 5>

<item 6>
Sunset Strip (2012 film)
Sunset Strip is a 2012 documentary directed by Hans Fjellestad, and produced by Tommy Alastra.

 The documentary explores the history of the mile and a half long stretch of road through West Hollywood known as Sunset Strip.
</item 6>

<item 7>
Mayor of the Sunset Strip
Mayor of the Sunset Strip is a 2003 documentary film on the life of Rodney Bingenheimer directed by George Hickenlooper, and produced by Chris Carter.

 In 2011, Craig Hlavaty of the "Houston Press" named "Mayor of the Sunset Strip" at number eight on the paper's list of "The 31 Best Music Documentaries of All
Time".

 The film won the Best Documentary Feature at the 2004 Santa Barbara International Film Festival.
</item 7>

<item 8>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 8>

<item 9>
Sunset Strip
Sunset Strip is the mile-and-a-half (2.4 km) stretch of Sunset Boulevard that passes through West Hollywood, California.

 It extends from West Hollywood's eastern border with Hollywood at Crescent Heights Boulevard to its western border with Beverly Hills at Sierra Drive.

 Sunset Strip is probably the best-known portion of Sunset Boulevard, with boutiques, restaurants, rock clubs, and nightclubs.

 It is also known for its array of huge, colorful billboards.
</item 9>

<item 10>
Riot on Sunset Strip
Riot on Sunset Strip is a 1967 counterculture-era exploitation movie, released by American International Pictures.

 It was filmed and released within four months of the late-1966 Sunset Strip curfew riots.
</item 10>
=== END ITEM id=5bdcc5432bd9bb0d ===

=== ITEM id=30885c4576dbad9a ===
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
The Preytells
The Preytells were an Australian indie rock band from Perth, formed in 2004 by Jessica Bennett (a.k.a. Audrey Tell) on lead guitar and backing vocals, Simon
Okely (a.k.a. Will Tell) on guitar and lead vocals, Jaclyn Pearson on drums and percussion and Cameron Stewart on bass guitar and backing vocals.

 Their sole album, "Flood Songs/June Songs", was issued in September 2009; before they disbanded later that year.

 Their single, "Shout!"

 ( May 2008), was nominated for WAM Song of the Year in the Pop category in 2008; while, "Lord Hold My Hand", was nominated for the same category in 2009.
</item 1>

<item 2>
Love Can't Wait (Nick Carter song)
"Love Can't Wait" is a song by American singer-songwriter Nick Carter, released as the lead Canadian single from his second studio album, "I'm Taking Off".

 The single was released on June 21, 2011.

 The song only appears on the Canadian version of the album, and has not been released in any other territories.

 The music video for the song features Carter performing the song with the backdrop of a motorway system.

 The music video also featured Canadian pop star Shawn Desman, who co-wrote the song.

 The single peaked at #2 on the Canadian Singles Chart, based entirely on download sales.

 Carter ran a competition for amateur mixers to remix the track, with the winning remixer earning $1500 and their remix appearing on Carter's remix album, "I'm
Taking Off: Relaunched and Remixed".
</item 2>

<item 3>
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
</item 3>

<item 4>
A Little Time
"A Little Time" is a song by The Beautiful South, and is the band's only single to reach number 1 in the UK Singles Chart.

 It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
</item 4>

<item 5>
I Told You So (Karmin song)
"I Told You So" is a song by American recording duo Karmin, taken from their debut EP "Hello".

 It was written by band members Amy Heidemann and Nick Noonan, as well as Elite and producer John "Jon Jon Traxx" Webb, Jr.

 The song features a primary hip hop influence, as well as elements of rock, electronica and ska.

 The song debuted live on "Saturday Night Live" on February 11, 2012.

 Heidemann and Noonan first performed a live acoustic snippet of the song on Elvis Duran and the Morning Show prior to their SNL appearance.

 "I Told You So" is said to pay homage to Chris Brown's "Look at Me Now", a song they covered on YouTube that brought them to fame.
</item 5>

<item 6>
Brand New Love Affair (song)
"Brand New Love Affair," sometimes alternatively listed as "Brand New Love Affair (Parts I and II)", is a song written by James Pankow for the group Chicago and
recorded for their album Chicago VIII.

 The song peaked at #61 on the charts.

 Guitarist Terry Kath sings the first half while bassist Peter Cetera sings the second half.

 Keyboardist Robert Lamm played the distinctive Fender Rhodes electric piano on the song—the intro particularly showcases its lush vibrato bell-like sound.
</item 6>

<item 7>
Zach Crowell
Zach Crowell is a Grammy Award-winning songwriter and record producer, who was born and raised in Nashville, Tennessee.

 Crowell is involved in a joint publishing deal with Warner/Chappell Music, Combustion Music and Taperoom Music.
</item 7>

<item 8>
Rebuild (song)
"Rebuild" is a song written by Jon Foreman of Switchfoot and Matt Thiessen of Relient K.

 The song was written in conjunction with the bands' 2007 Appetite for Construction Tour, featuring members from all three bands on tour, Switchfoot, Relient K,
and Ruth.

 Thiessen sang second vocals, and Dustin Ruth of Ruth played harmonica.
</item 8>

<item 9>
Tha Trademarc
Marc Predka (born April 21, 1975) better known by his stage name Tha Trademarc, is an American hip hop artist.

 He first came to prominence when he and his first cousin, WWE wrestler John Cena, collaborated on the 2005 album "You Can't See Me", including Cena's entrance
theme "The Time Is Now".

 He also appeared in the music videos of the songs, "Bad Bad Man" and "Right Now" with Cena.

 Prior to the album, Trademarc appeared on Cena's first WWE DVD "Word Life".

 He now works as a classroom specialist for the Peace Corps of America.
</item 9>

<item 10>
The Time Is Now (John Cena song)
"The Time Is Now" is a song by professional wrestler, actor, and hip hop musician John Cena.

 It is the current entrance theme Cena uses as he makes his entrance to the ring.

 Currently wrestling for WWE, Cena recorded the song in 2005 for his debut studio album, "You Can't See Me".
</item 10>
=== END ITEM id=30885c4576dbad9a ===

=== ITEM id=1c2fc0ae28ff5500 ===
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
Lydia Patterson Institute
Lydia Patterson Institute is a Methodist Christian college-preparatory school located in El Paso, Texas, United States.

 Founded in 1913, it offers programs for Spanish-speaking children, primarily from Juarez, to attend high school in the United States and attend an American
undergraduate university.

 All high school classes are taught in English, and the school is accredited by the Southern Association of Colleges and Schools.
</item 1>

<item 2>
Graph query row:
organization: Presbyterian University College, Ghana
founded_on: 23 November 2003
relationship: LOCATED_IN
connected_entity: Abetifi-Kwahu
</item 2>

<item 3>
Syracuse Triad
The Syracuse Triad is the name given to the three women's sororities founded at Syracuse University.

 Alpha Phi was founded first in 1872 by 10 of the original 20 women admitted into Syracuse University.

 Gamma Phi Beta came along two years later in 1874 and with it came the term "sorority," which was coined at the time of its founding.

 (Prior to that, women's Greek-letter organizations used the term "women's fraternity," since no more appropriate term existed.)

 Alpha Gamma Delta completed the triad in 1904.
</item 3>

<item 4>
Evangelical Presbyterian University College
The Evangelical Presbyterian University College was established in 2008 by the Evangelical Presbyterian Church, Ghana.
</item 4>

<item 5>
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
</item 5>

<item 6>
Presbyterian University College
The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.

 It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.

 It was established by the Presbyterian Church of Ghana on 23 November 2003 and inaugurated on 27 March 2004 by the president of Ghana, John Agyekum Kufuor.
</item 6>

<item 7>
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
</item 7>

<item 8>
Aaron McCreary
Aaron Monroe "Mac" McCreary (September 15, 1892 – ?)

 was an American football, basketball, and baseball player, coach, and college athletics administrator.

 He served as the head football coach at Tempe State Teachers College, known at Arizona State Teachers College by 1929 and now called Arizona State University,
compiling a career college football record of 25–17–4.

 McCreary was also the head basketball coach at Tempe/Arizona State Teachers from 1923 to 1930 and at Arizona State Teacher's College of Flagstaff, now Northern
Arizona University, amassing a career college basketball record of 140–149.

 In addition, he coached baseball at Tempe/Arizona State Teachers (1924–1926, 1928, 1930–1931) and at Arizona State Teacher's Flagstaff in 1959, tallying a
career college baseball mark of 22–56–1.
</item 8>

<item 9>
Sanders–Brown Center on Aging
The Sanders–Brown Center on Aging at the University of Kentucky in Lexington, Kentucky, is responsible for research, education and service programs in aging.

 The program was founded in 1963 with the creation of the Council on Aging.

 In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.

 The center was named for the Browns and his then-business partner, (Col.) Harlan Sanders.

 The four-story brick and concrete structure fronts South Limestone at the edge of the medical campus.

 The complex features 32 research laboratories that employs more than 150.
</item 9>

<item 10>
Graph query row:
organization: The University of Arizona
founded_on: 1885
relationship: LOCATED_IN
connected_entity: Tucson
</item 10>
=== END ITEM id=1c2fc0ae28ff5500 ===

=== ITEM id=46da91c09ff0db85 ===
Gold sentences and facts:
Sentence 0: Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National Remembrance.
  Fact 0.0: Kolejka is a board game.
  Fact 0.1: Kolejka is Polish for 'queue' or 'line (of people)'.
  Fact 0.2: Karol Madaj designed Kolejka.
  Fact 0.3: Kolejka was launched in February 2011.
  Fact 0.4: Kolejka was launched by Poland's Institute of National Remembrance.
Sentence 1: Ghettopoly is a parody of "Monopoly" released in 2003.
  Fact 1.0: Ghettopoly is a parody of Monopoly.
  Fact 1.1: Ghettopoly was released in 2003.

Retrieved items:
<item 1>
The White EP (Vib Gyor EP)
The "White EP" is a promo EP from the Leeds band Vib Gyor.

 It was first sold on their UK tour with American band Transfer and was released through their My Space page and official web-site on 13 November 2007.

 It is the follow-up to their The Secret EP.

 It was recorded in Rockfield studios in South Wales and produced by Ken Thomas.
</item 1>

<item 2>
Kick Over the Traces
Kick Over the Traces is a compilation album by the American punk rock band Pinhead Gunpowder.

 The album contains tracks from Pinhead Gunpowder releases since the band formed in 1990.

 The album was released through Recess Records on June 16, 2009.

 The Japanese edition features a second disc recorded live at 924 Gilman Street on February 10, 2008, and features different cover art.
</item 2>

<item 3>
Compulsive Disclosure
Compulsive Disclosure is the second compilation album by the American punk rock band Pinhead Gunpowder.

 It was released on October 21, 2003, through Lookout!

 Records.

 The album features songs from the group's eponymously titled 2000 EP, the "Dillinger Four / Pinhead Gunpowder" split EP, the "8 Chords, 328 Words" EP and also
features re-recorded versions of the songs "2nd Street" and "At Your Funeral" (originally from "Dillinger Four / Pinhead Gunpowder").

 "Compulsive Disclosure" was re-released on CD and vinyl through Recess Records on February 12, 2010, with two unreleased tracks, "Salting Agents" and "El Lasso
Grappo".
</item 3>

<item 4>
Pinhead Gunpowder (2008 EP)
Pinhead Gunpowder (also known by fans as the West Side Highway EP) is the second eponymously titled extended play by the American punk rock band Pinhead
Gunpowder.

 It was released on August 19, 2008, through Recess Records, the group's first release on Recess.

 "Pinhead Gunpowder" marked the band's first new release in over 5 years.

 The first pressings of the extended play were on green and blue vinyl.

 All of the tracks were rereleased on the 2009 compilation "Kick Over the Traces".
</item 4>

<item 5>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 5>

<item 6>
The First Collage
The First Collage is the debut and first EP by Yang Yo-seob, released on November 26, 2012.
</item 6>

<item 7>
OK Go (2000 EP)
OK Go or The Brown EP (officially known as OKGoCD.001) is the first EP by American rock band OK Go.
</item 7>

<item 8>
The Hunter (EP)
The Hunter is the first EP (and follow up to the 2010 solo album "The Boxer") by Kele Okereke (under the professional name Kele), lead singer and rhythm
guitarist of the British rock band Bloc Party.

 It was released on 7 November 2011 by Wichita Recordings in the UK, set back a week from the original release date, and 3 days earlier on 4 November 2011 by
Wichita Recordings and Liberator Music in Australia.

 The first single released from the EP was "What Did I Do?

", which features guest vocals from Lucy Taylor.

 The music video was released on 13 September 2011, with the song released as a digital single in Japan 10 days later.
</item 8>

<item 9>
OK Go (2001 EP)
OK Go or The Pink EP (officially known as OKGoCD.002) is the second EP by American rock band OK Go.
</item 9>

<item 10>
Ghettopoly
Ghettopoly is a parody of "Monopoly" released in 2003.

 Invented by David Chang, it uses "Monopoly"-like mechanics in the atmosphere of a caricaturized United States ghetto.
</item 10>
=== END ITEM id=46da91c09ff0db85 ===

=== ITEM id=27eea1cf2f32c92d ===
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
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 1>

<item 2>
Heavyweights
Heavyweights is a 1995 American comedy-drama film directed by Steven Brill and written by Brill with Judd Apatow.

 The film centers around a fat camp for kids that is taken over by a fitness guru named Tony Perkis (Ben Stiller).
</item 2>

<item 3>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 3>

<item 4>
The Apple Dumpling Gang Rides Again
The Apple Dumpling Gang Rides Again is a 1979 American comedy-western film produced by Walt Disney Productions and a sequel to "The Apple Dumpling Gang" (1975),
starring the comedy duo of Tim Conway and Don Knotts reprising their respective roles as Amos and Theodore.

 The film also stars Tim Matheson, Harry Morgan, and Kenneth Mars.

 Ruth Buzzi appears in a small cameo as a wild farsighted woman.

 Robert Totten, who directed installments of "Gunsmoke", also had a small part in the film.
</item 4>

<item 5>
The Sasquatch Gang
The Sasquatch Gang (also known as The Sasquatch Dumpling Gang) is a 2006 comedy film written and directed by Tim Skousen, the first assistant director on
"Napoleon Dynamite".
</item 5>

<item 6>
Honey, I Blew Up the Kid
Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".

 Directed by Randal Kleiser and released by Walt Disney Pictures, the film stars Rick Moranis, Marcia Strassman, Robert Oliveri and Amy O'Neill, who reprise
their roles as Wayne, Diane, Nick, and Amy Szalinski respectively, as well as newcomer Keri Russell as Mandy Park, Nick's love interest and babysitter of Adam,
the Szalinskis' new two-year-old son, whose accidental exposure to Wayne's new industrial-sized growth machine causes him to gradually grow to enormous size.

 Made only three years after "Honey, I Shrunk the Kids", this film is set five years after the events depicted in the previous film.

 It was filmed in 1991.
</item 6>

<item 7>
The Apple Dumpling Gang
The Apple Dumpling Gang is a 1971 novel by Jack Bickham, about a group of orphaned children during the California gold rush.

 They encounter a gambler who reluctantly helps them, as well as a pair of hapless robbers who are after the gold the children have found.
</item 7>

<item 8>
Buddy Baker (composer)
Norman Dale "Buddy" Baker (January 4, 1918 – July 26, 2002) was an American composer who, together with Paul J. Smith, scored many Disney films, such as "The
Apple Dumpling Gang" in 1975, "The Apple Dumpling Gang Rides Again" in 1979, "The Shaggy D.A." in 1976, "The Many Adventures of Winnie the Pooh" in 1977, and
"The Fox and the Hound" in 1981.
</item 8>

<item 9>
The Apple Dumpling Gang (film)
The Apple Dumpling Gang is a 1975 American comedy-western film produced by Walt Disney Productions about a slick gambler named Russell Donovan (Bill Bixby) who
is duped into taking care of a group of orphans who eventually strike gold during the California Gold Rush.
</item 9>

<item 10>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 10>
=== END ITEM id=27eea1cf2f32c92d ===

=== ITEM id=0f7a9f020810a71f ===
Gold sentences and facts:
Sentence 0:  He appeared in 57 films between 1919 and 1932.
  Fact 0.0: The person described appeared in 57 films.
  Fact 0.1: The person described appeared in 57 films between 1919 and 1932.
Sentence 1:  He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
  Fact 1.0: The person described directed 54 films.
  Fact 1.1: The person described directed these films between 1920 and 1933.
  Fact 1.2: The person's directed films include one Harry Langdon short.
  Fact 1.3: The person's directed films include two of the Tay Garnett-penned comedies Stan Laurel made for Joe Rock.
  Fact 1.4: The person's directed films include fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
Sentence 2: John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.
  Fact 2.0: John Scott Biddle lived from 1925 to 2008.
  Fact 2.1: John Scott Biddle was a foremost yachting cinematographer.
  Fact 2.2: John Scott Biddle was a lecturer.
  Fact 2.3: John Scott Biddle established a film-making career.
  Fact 2.4: John Scott Biddle's film-making career spanned more than forty years.

Retrieved items:
<item 1>
Millard Webb
Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.

 He directed 20 films between 1920 and 1933.

 His best-known film is the 1926 silent John Barrymore adventure "The Sea Beast" costarring Dolores Costello.

 Webb also directed the early sound Florenz Ziegfeld produced talkie "Glorifying the American Girl" released by Paramount in 1929.

 His active years were from 1916 to 1933.
</item 1>

<item 2>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 2>

<item 3>
Harry Sweet
Harry Sweet (October 2, 1901 – June 18, 1933) was an American actor, director and screenwriter.

 He appeared in 57 films between 1919 and 1932.

 He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and
fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
</item 3>

<item 4>
Charles John Biddle
Charles John Biddle (April 30, 1819 – September 28, 1873) was an American soldier, lawyer, congressman, and newspaper editor.
</item 4>

<item 5>
John Biddle (United States Army general)
John Biddle (February 2, 1859 – January 18, 1936) was career United States Army officer who became superintendent of the United States Military Academy.
</item 5>

<item 6>
Jack Bickham
Jack Miles Bickham (September 2, 1930 – July 25, 1997) was an American author who wrote 75 published novels, of which two were made into movies, "The Apple
Dumpling Gang" and "Baker's Hawk".
</item 6>

<item 7>
Charles J. Biddle (aviator)
Major Charles John Biddle (13 May 1890 – 22 March 1972) was an American aviator, attorney, and author.

 He was a flying ace during World War I.

 Postwar, he launched a career in law and wrote his memoirs.
</item 7>

<item 8>
Biddle family
The American Biddle family of Philadelphia began when William Biddle (1630–1712) and Sarah Kempe (1634–1709), Quakers, emigrated from England to America in
1681, in part, to avoid religious persecution.

 Having acquired extensive rights to over 43000 acre of lands in Quaker West Jersey, they settled first at Burlington.

 Two third generation brothers, William Biddle, 3rd (1698–1756), and John Biddle (1707–1789), moved from Mount Hope (1684) near Bordentown to Philadelphia in
the 1720s and 1730s and constituted the first generation of the Philadelphia Biddle family.
</item 8>

<item 9>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 9>

<item 10>
John Biddle (Michigan politician)
John Biddle (March 2, 1792 – August 25, 1859) was a delegate to the United States Congress from the Michigan Territory.
</item 10>
=== END ITEM id=0f7a9f020810a71f ===

=== ITEM id=8d3a6fd13132b24d ===
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
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 1>

<item 2>
The entries cover historic U.S. canals, waterways, water systems, and sites. Transport canals include Beaver–Erie (136 miles), Delaware (60), Whitewater (76),
and Susquehanna–Tidewater (43), plus interstate and regional routes; many declined with railroad competition. The Dismal Swamp Canal, opened in 1805 and closed
in 2016, served the Intracoastal Waterway; nearby park and refuge protect the swamp.
</item 2>

<item 3>
Genesee River
The Genesee River is a tributary of Lake Ontario flowing northward through the Twin Tiers of Pennsylvania and New York in the United States.
</item 3>

<item 4>
New York’s Erie Canal heritage includes Lock 52 (1849–53), canal buildings and archaeological sites; the 1836–42 Broad Street Aqueduct later became a road and
subway route. The 1905 song “Low Bridge” recalls mule-powered canal traffic. The 525-mile state canal system includes the Erie, Oswego, Cayuga–Seneca and
Champlain canals. The Genesee River and tributaries Black Creek and Oatka Creek drain western New
</item 4>

<item 5>
Genesee Valley Park
Genesee Valley Park is located in the south side of Rochester, New York along the shores of the Genesee River.

 The New York State Barge Canal (the currently in-use portion of the Erie Canal) crosses the Genesee River within the park.

 The University of Rochester is located near the park's north entrance.
</item 5>

<item 6>
Tamiami Canal
The Tamiami Canal or C-4 Canal, is a canal located in southern Florida in the United States.

 It flows in a west to east direction from the western part of the state in the Everglades past the Miami International Airport to a salinity control center
near the Miami River.

 It averages 8 ft in depth and is over 100 ft wide in some areas.
</item 6>

<item 7>
Black Creek (Genesee River)
Black Creek is a tributary of the Genesee River that runs for 46 mi in western New York, United States.

 The creek begins in Middlebury in Wyoming County and runs north for roughly the first half of its course and east for the other half, eventually joining the
Genesee River in Chili, Monroe County.

 Its drainage area spans around 202 mi2 , which is largely rural and agricultural.

 As of 2000, 40,000 residents lived in the creek's drainage area.
</item 7>

<item 8>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 8>

<item 9>
New York State Canal Corporation
The New York State Canal Corporation is responsible for the oversight, administration and maintenance of the New York State Canal System, which consists of the
Erie Canal, Cayuga–Seneca Canal, Oswego Canal and Champlain Canal.

 It is also involved with the development and maintenance of the New York State Canalway Trail and with the general development and promotion of the Erie Canal
Corridor as both a tourist attraction and a working waterway.
</item 9>

<item 10>
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 10>
=== END ITEM id=8d3a6fd13132b24d ===
