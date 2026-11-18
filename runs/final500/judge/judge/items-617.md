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

=== ITEM id=ac270be4c2f8bffe ===
Gold sentences and facts:
Sentence 0: The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
  Fact 0.0: The Second Battle of Guam was fought from 21 July to 10 August 1944.
  Fact 0.1: The Second Battle of Guam was the American recapture of the Japanese-held island of Guam.
  Fact 0.2: Guam is a U.S. territory in the Mariana Islands.
  Fact 0.3: Japan captured Guam from the U.S. in the 1941 First Battle of Guam.
  Fact 0.4: The Second Battle of Guam took place during the Pacific campaign of World War II.
Sentence 1: The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and Antietam Creek as part of the Maryland Campaign.
  Fact 1.0: The Battle of Antietam was also known as the Battle of Sharpsburg, particularly in the South.
  Fact 1.1: The Battle of Antietam was fought on September 17, 1862.
  Fact 1.2: The Battle of Antietam was fought near Sharpsburg, Maryland.
  Fact 1.3: The Battle of Antietam was fought near Antietam Creek.
  Fact 1.4: The Battle of Antietam was part of the Maryland Campaign.

Retrieved items:
<item 1>
Seven Days Confederate order of battle
The following Confederate States Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization during the battle, the casualty returns and the reports.

 The Union order of battle is listed separately.
</item 1>

<item 2>
Seven Days Union order of battle
The following Union Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization-return of casualties during the battle and the reports.

 The Confederate order of battle is listed separately.
</item 2>

<item 3>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 3>

<item 4>
Battle of Gaines's Mill
The Battle of Gaines's Mill, sometimes known as the First Battle of Cold Harbor or the Battle of Chickahominy River, took place on June 27, 1862, in Hanover
County, Virginia, as the third of the Seven Days Battles (Peninsula Campaign) of the American Civil War.

 Following the inconclusive Battle of Beaver Dam Creek (Mechanicsville) the previous day, Confederate General Robert E. Lee renewed his attacks against the
right flank of the Union Army, relatively isolated on the northern side of the Chickahominy River.

 There, Brig. Gen. Fitz John Porter's V Corps had established a strong defensive line behind Boatswain's Swamp.

 Lee's force was destined to launch the largest Confederate attack of the war, about 57,000 men in six divisions.

 Porter's reinforced V Corps held fast for the afternoon as the Confederates attacked in a disjointed manner, first with the division of Maj. Gen. A.P. Hill,
then Maj. Gen. Richard S. Ewell, suffering heavy casualties.

 The arrival of Maj. Gen. Stonewall Jackson's command was delayed, preventing the full concentration of Confederate force before Porter received some
reinforcements from the VI Corps.
</item 4>

<item 5>
Battle of Antietam
The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and
Antietam Creek as part of the Maryland Campaign.

 It was the first field army–level engagement in the Eastern Theater of the American Civil War to take place on Union soil and is the bloodiest single-day
battle in American history, with a combined tally of 22,717 dead, wounded, or missing.
</item 5>

<item 6>
George McMillin
George Johnson McMillin (November 25, 1889 – August 29, 1983 ) was a United States Navy Rear Admiral who served as the 38th and final Naval Governor of Guam.

 He graduated from the United States Naval Academy in 1911 and served as an officer during four separate conflicts: World War I, the occupation of the Dominican
Republic, the United States occupation of Veracruz, and World War II.

 He served on the staff of both the Naval Academy and the Naval War College as well.

 He is most remembered as the commander who surrendered Guamanian forces to a much larger Japanese force during the First Battle of Guam, only the second battle
of World War II involving the United States.

 He had previously evacuated all but one civilian American citizen from the island and attempted to rebuild defenses after a strong typhoon devastated the
island the year before.

 On December 8, 1941, Japanese forces invaded Guam and McMillin surrendered two days later.

 He spent the rest of the war at various Japanese prisoner of war camps.
</item 6>

<item 7>
Seven Days Battles
The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil
War.

 Confederate General Robert E. Lee drove the invading Union Army of the Potomac, commanded by Maj. Gen. George B. McClellan, away from Richmond and into a
retreat down the Virginia Peninsula.

 The series of battles is sometimes known erroneously as the Seven Days Campaign, but it was actually the culmination of the Peninsula Campaign, not a separate
campaign in its own right.
</item 7>

<item 8>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 8>

<item 9>
Maryland Campaign
The Maryland Campaign—or Antietam Campaign—occurred September 4–20, 1862, during the American Civil War.

 Confederate Gen. Robert E. Lee's first invasion of the North was repulsed by the Army of the Potomac under Maj. Gen. George B. McClellan, who moved to
intercept Lee and his Army of Northern Virginia and eventually attacked it near Sharpsburg, Maryland.

 The resulting Battle of Antietam was the bloodiest single-day battle in American history and is widely considered one of the major turning points of the war.
</item 9>

<item 10>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 10>
=== END ITEM id=ac270be4c2f8bffe ===

=== ITEM id=a1a499f2c7d9d7c5 ===
Gold sentences and facts:
Sentence 0:  He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.
  Fact 0.0: The person finished second in voting for the Heisman Trophy in 1956.
  Fact 0.1: Paul Hornung finished first in that Heisman Trophy voting in 1956.
Sentence 1: Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.
  Fact 1.0: Paul Vernon Hornung was born on December 23, 1935.
  Fact 1.1: Paul Vernon Hornung is nicknamed "The Golden Boy".
  Fact 1.2: Paul Vernon Hornung is a former professional American football player.
  Fact 1.3: Paul Vernon Hornung is a Hall of Fame running back.
  Fact 1.4: Paul Vernon Hornung played for the Green Bay Packers of the National Football League (NFL).
  Fact 1.5: Paul Vernon Hornung played for the Green Bay Packers from 1957 to 1966.
  Fact 1.6: Paul Vernon Hornung won four NFL titles.
  Fact 1.7: Paul Vernon Hornung won the first Super Bowl.

Retrieved items:
<item 1>
Johnny Majors
John Terrill Majors (born May 21, 1935) is a former American football player and coach.

 A standout halfback at the University of Tennessee, he was an All-American in 1956 and a two-time winner of the Southeastern Conference Most Valuable Player
award, in 1955 and 1956.

 He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.

 Majors served as the head football coach at Iowa State University (1968–1972), the University of Pittsburgh (1973–1976, 1993–1996), and Tennessee (1977–1992),
compiling a career college football record of 185–137–10.

 His 1976 Pittsburgh squad won a national championship after capping a 12–0 season with a victory in the Sugar Bowl.

 Majors was inducted into the College Football Hall of Fame as a player in 1987.
</item 1>

<item 2>
1997 Tennessee Volunteers football team
However, the Vols' defense could not stop Nebraska's rushing attack, giving up more than 400 yards on the ground in a 42–17 loss.

 As a senior, Manning won numerous awards.

 He was a consensus first-team All-American and won the Maxwell Award, the Davey O'Brien Award, the Johnny Unitas Award, and the Best College Football Player
ESPY Award, among others.

 However, he did not win the Heisman Trophy, finishing runner-up to Charles Woodson, a CB from Michigan, and the only defensive player ever to win the Heisman
Trophy.
</item 2>

<item 3>
Johnny Bright
Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.

 He played college football at Drake University.

 He is a member of the Canadian Football Hall of Fame, the National Football Foundation's College Football Hall of Fame, the Missouri Valley Conference Hall of
Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame, and the "Des Moines Register's" Iowa Sports Hall of Fame.
</item 3>

<item 4>
2012 Texas A&amp;M vs. Alabama football game
The 2012 Texas A&M vs. Alabama football game was a college football game between the Texas A&M Aggies and Alabama Crimson Tide at Bryant–Denny Stadium in
Tuscaloosa, Alabama.

 This was the game where Texas A&M freshman quarterback and eventual Heisman Trophy winner Johnny Manziel completed 24 of 31 passes, had 253 passing yards, and
two passing touchdowns with 92 rushing yards to help No. 15 Texas A&M upset No. 1 Alabama 29–24, which led him to being the first freshman to win the Heisman
Trophy.
</item 4>

<item 5>
1976 NCAA Division I football season
The 1976 NCAA Division I football season ended with a championship for the Panthers of the University of Pittsburgh.

 Led by head coach Johnny Majors (voted the AFCA Coach of the Year), the Pitt Panthers brought a college football championship to the home of the defending pro
football champions, the Steelers.

 Pitt also had the Heisman Trophy winner, Tony Dorsett; the Panthers had been ranked ninth in the preseason AP poll.
</item 5>

<item 6>
Heisman curse
Although many Heisman winners have not enjoyed success at the professional level, including players like Matt Leinart, Andre Ware, Jason White, Rashaan Salaam,
Eric Crouch, Ty Detmer, Troy Smith and Gino Torretta, proponents of the "curse" rarely cite highly successful players such as Barry Sanders, Charles Woodson,
Eddie George, Tim Brown, Bo Jackson, Marcus Allen, Earl Campbell, and Tony Dorsett among the notables.
</item 6>

<item 7>
Frank Eliscu
Frank Eliscu (1912–1996) was an American sculptor and art teacher who designed and created the Heisman Memorial Football Trophy in 1935 when he was only 20
years old.

 The first Heisman Trophy, a strong young bull of a football player cast in bronze, was presented to a college football player in 1935, and is considered one of
the greatest honors a college athlete can receive.

 Over the years the Heisman Committee has paid tribute to Eliscu and his creation several times, and in 1985, the Heisman Committee invited him to speak at the
50th Anniversary Heisman ceremonies.
</item 7>

<item 8>
Heisman curse
The Heisman curse is a term coined to reference a two-part assertion of a negative future for the winning player of the Heisman Trophy.

 The "curse" supposes that any college football player who wins the Heisman plays on a team that will likely lose its subsequent bowl game.

 The trend of post-award failure has garnered the attention of the mainstream media.

 Talk of a curse in relation to bowl results was particularly prevalent from 2003 to 2008, when six Heisman Trophy winners compiled a cumulative 1–5 bowl game
record, and five of those six led number one ranked teams into the Bowl Championship Series (BCS) National Championship Game as favorites (Heisman Trophy
winners, including Reggie Bush, who gave back his Heisman Trophy, are 4–8 overall in the BCS National Championship Game and College Football Playoff National
Championship, although prior to 2009 they were 1–6).

 Additionally, the Heisman curse asserts that in most cases a Heisman winner will have either a poor career in the National Football League (NFL), or in fact
not even see such a football career at all.
</item 8>

<item 9>
Heisman Trophy
The Heisman Memorial Trophy (usually known colloquially as the Heisman Trophy or The Heisman), is awarded annually to the most outstanding player in college
football in the United States whose performance best exhibits the pursuit of excellence with integrity.

 Winners epitomize great ability combined with diligence, perseverance, and hard work.

 It is presented by the Heisman Trophy Trust in early December before the postseason bowl games.
</item 9>

<item 10>
Paul Hornung
Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the
Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.

 He is the first in pro football history to win the Heisman Trophy, be selected as the first overall selection in the NFL Draft, win the NFL most valuable
player award, and be inducted into both the professional and college football halls of fame.
</item 10>
=== END ITEM id=a1a499f2c7d9d7c5 ===

=== ITEM id=a7d9cc87c34eb341 ===
Gold sentences and facts:
Sentence 0: Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
  Fact 0.0: Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book.
  Fact 0.1: Lawrence Weschler wrote Mr. Wilson's Cabinet of Wonder.
  Fact 0.2: Mr. Wilson's Cabinet of Wonder is primarily about the Museum of Jurassic Technology.
  Fact 0.3: Mr. Wilson's Cabinet of Wonder is more broadly about the history and role of museums.
Sentence 1: The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.
  Fact 1.0: The Museum of Jurassic Technology is a museum.
  Fact 1.1: The Museum of Jurassic Technology is located at 9341 Venice Boulevard.
  Fact 1.2: The Museum of Jurassic Technology is in the Palms district of Los Angeles, California.

Retrieved items:
<item 1>
Museum of Jurassic Technology
The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.

 It was founded by David Hildebrand Wilson and Diana Drake Wilson (husband and wife) in 1988.
</item 1>

<item 2>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 2>

<item 3>
Gate of Ivory, Gate of Horn
Gate of Ivory, Gate of Horn is a fantasy novel by British author Robert Holdstock.

 It was originally published in the United States in 1997 (and in the United Kingdom under the title "Gate of Ivory" in 1998.)

 The story is a prequel to "Mythago Wood" and explores Christian Huxley's quest into Ryhope Wood and the apparent suicide of his mother, Jennifer Huxley.

 The title of the book refers to the gates of horn and ivory described in both Homer's "Odyssey" and Virgil's "Aeneid".
</item 3>

<item 4>
The Lost City (Dungeons &amp; Dragons)
The Lost City (B4) is a "Dungeons & Dragons" adventure module by Tom Moldvay.

 It was first published by TSR in 1982 and was designed as a stand-alone adventure for use with the "Dungeons & Dragons Basic Set".

 The working title for the module was "The Lost City of Cynidecia".

 Moldvay designed the module to give novice Dungeon Masters experience fleshing out adventures and is only partially complete.

 The module is described as a low-level scenario, in which the only hope of the player characters' survival can be found in a ruined city slowly rising out of
the sands.

 The adventure is set inside a huge step pyramid, with the lower pyramid only sketched out and the city itself described with a list of the major areas and a
map.

 The adventure’s main villain is Zargon, a giant one-eyed monster and his minions.

 The entire double pyramid, not including the city, contains over 100 rooms.
</item 4>

<item 5>
Mr. Wilson's Cabinet of Wonder
Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily
about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
</item 5>

<item 6>
Primeval Whirl
Primeval Whirl is a steel wild mouse roller coaster at Disney's Animal Kingdom at the Walt Disney World Resort.

 The ride is a spinning roller coaster purchased from Reverchon Industries.

 The ride is part of Chester and Hester's Dino-Rama, which is itself part of DinoLand U.S.A. It is a roller coaster in the "Mild But Wild Thrills" category.

 It has cars that spin in circles while traveling on tracks, permitting the ride experience to vary greatly each time it is ridden.

 The ride features 13 cars, each seating up to 4 riders.
</item 6>

<item 7>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 7>

<item 8>
John G. Wilson
John Guy Wilson (1842–1892) was a Manchester, UK patent agent and the founder of Wilson Gunn, one of the oldest firms of Patent and Trade Mark Attorneys in the
UK.

 John G. Wilson built up the business until his death in the 1890s.

 As a side line he acted as agent to the Alliance Insurance Company.

 There is no record that Mr. Wilson ever became a member of the Chartered Institute of Patent Agents.
</item 8>

<item 9>
Merlin's Wood
Merlin's Wood; or, The Vision of Magic is a short novel by British writer Robert Holdstock, first published in the United Kingdom in 1994.

 The novel is considered part of the "Mythago Wood" cycle, but takes place in Brittany, France instead of Herefordshire, England.

 The work has all new characters and focuses on the mythical birthplace and burial site of Merlin, the magical wood Brocéliande.

 Brocéliande is a smaller version of Ryhope wood where British myth predominates.
</item 9>

<item 10>
Witmer Stone
Witmer Stone (September 22, 1866 – May 24, 1939) was an American ornithologist, botanist, and mammalogist, and was considered one of the last of the “great
naturalists.”

 Stone is remembered principally as an ornithologist.

 He was president of the American Ornithologists’ Union (AOU) 1920–23, and was editor of the AOU’s periodical "The Auk" 1912–1936.

 He spearheaded the production of the 4th edition of the AOU checklist, published in 1931.

 He worked for over 50 years in the Ornithology Department at the Academy of Natural Sciences of Philadelphia, eventually serving as Director of the
institution.

 Stone was one of the founding members of the Delaware Valley Ornithological Club (DVOC) in 1890 and was actively involved in the organization for the remainder
of his life.

 Stone was one of only two scientists (Joseph Grinnell was the other) to serve as president of both the AOU and the American Society of Mammalogists, and he
co-authored two popular books about mammals.
</item 10>
=== END ITEM id=a7d9cc87c34eb341 ===

=== ITEM id=aa8abce28aaa5fe8 ===
Gold sentences and facts:
Sentence 0: Mirpur University of Science & Technology (میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی) (MUST) was formerly a constituent college of University of Azad Jammu and Kashmir as University College of Engineering & Technology Mirpur (UCET Mirpur).
  Fact 0.0: Mirpur University of Science & Technology (MUST) was formerly a constituent college of University of Azad Jammu and Kashmir.
  Fact 0.1: The Urdu name of Mirpur University of Science & Technology is میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی.
  Fact 0.2: Mirpur University of Science & Technology was formerly named University College of Engineering & Technology Mirpur (UCET Mirpur).
Sentence 1: The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.
  Fact 1.0: The University of Debrecen is a university located in Debrecen, Hungary.
  Fact 1.1: The Hungarian name of the University of Debrecen is 'Debreceni Egyetem'.

Retrieved items:
<item 1>
Béla Bartók Music High School
Béla Bartók Music High School (Bartók Béla Zeneművészeti Szakközépiskola) is situated in the Palace of Music (Zenepalota) in Bartók square Miskolc, Hungary.

It is a famous Music school named after the famous Hungarian composer Béla Bartók.

It was founded in 1966.
</item 1>

<item 2>
Bangladesh University of Business and Technology
Bangladesh University of Business and Technology (Bengali: বাংলাদেশ ইউনিভার্সিটি অফ বিজনেস অ্যান্ড টেকনোলজি ) or BUBT is one of the eight private universities
to receive the green Signal from the government, located in Mirpur, Dhaka, Bangladesh.

 The university was established under the Private University Act 1992.

 BUBT is regulated by the Bangladesh University Grants Commission (UGC).
</item 2>

<item 3>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 3>

<item 4>
Mirpur University of Science and Technology
Mirpur University of Science & Technology (میرپور یونیورسٹی براۓ سائنس اور ٹیکنولوجی) (MUST) was formerly a constituent college of University of Azad Jammu and
Kashmir as University College of Engineering & Technology Mirpur (UCET Mirpur).

 It is a state university and the President of Azad Jammu & Kashmir is the Chancellor of the university.

 The Vice-Chancellor is the executive head and manages the university functions.
</item 4>

<item 5>
Education in Hyderabad
Hyderabad is an important seat of learning in southern India.

 The city hosts two central universities, three deemed universities, and six state universities.

 Osmania University is one of the oldest universities in India.

 Many institutes for education like University of Hyderabad, Jawaharlal Nehru Technological University, Hyderabad, International Institute of Information
Technology, Indian Institute of Technology, Hyderabad, Tata Institute of Fundamental Research,Hyderabad, Icfai University, Birla Institute of Technology and
Science, Pilani – Hyderabad and medical colleges like Nizam's Institute of Medical Sciences are located there.

 Also based in the city are the Institute of Public Enterprise and the National Academy of Legal Studies & Research (NALSAR).

 Hyderabad has various research institutes such as the Indian Institute of Chemical Technology, Centre for Cellular and Molecular Biology.

 It is also the home of Maulana Azad National Urdu University as well as Dr. B.R. Ambedkar Open University.

 This educational infrastructure attracts students from all over India and some international students (especially from Africa and the Middle East).

 Most prolific business school in India Nmims is also expanding its roots in Hyderabad.
</item 5>

<item 6>
University of Debrecen
The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.

 It is the oldest continuously operating institution of higher education in Hungary (since 1538).

 The university has a well established programme in the English language for international students, particularly in the Medical field, which first established
education in English in 1986.

 There are nearly 4000 international students studying at the university.
</item 6>

<item 7>
Albanian University in Berat
Through presentations, research and exchange, the University directs and encourages the identification and development of projects that will once again
consolidate its primary purpose, scientific development, and research toward better quality.
</item 7>

<item 8>
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
</item 8>

<item 9>
Palace of Music (Miskolc)
The Palace of Music ("Zenepalota") is a building in Bartók square, Miskolc, Hungary.

 It is the building of the Béla Bartók Secondary School and the Béla Bartók Music Institute (a faculty of the University of Miskolc.)

 The Palace was designed by Gyula Waelder in Neo-baroque style and was built between 1926 and 1927.

 The construction was financed from USA loans, just like that of the Hotel Palace in Lillafüred and the Market Hall on Búza tér.
</item 9>

<item 10>
Ural Federal University
The Ural Federal University named after the first President of Russia B. N. Yeltsin (Russian: Уральский федеральный университет имени первого Президента России
Б.Н.

 Ельцина, "Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina", often shortened to UrFU, УрФУ ) (formed by a merger of the Ural
State Technical University and Ural State University) is one of the leading educational institutions in the Ural region.

 Ural Federal University acts as a research and innovation center of the Ural region and has close cooperation with the Russian Academy of Sciences.

 Training of students is carried out in four main areas of knowledge and 108 academic majors.

 The links between the university and intermediate educational institutions are actively developed.
</item 10>
=== END ITEM id=aa8abce28aaa5fe8 ===

=== ITEM id=cd03a1aed3e68f6d ===
Gold sentences and facts:
Sentence 0: The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
  Fact 0.0: The Albanian Coalition 'Perspective' is an Albanian political party.
  Fact 0.1: The Albanian Coalition 'Perspective' is in Montenegro.
Sentence 1: Albanians in Montenegro (, Albanian: "Shqiptarët e Malit të Zi" ) constitute 4.91% of the county's total population.
  Fact 1.0: Albanians in Montenegro constitute 4.91% of the county's total population.
  Fact 1.1: The Albanian name for Albanians in Montenegro is 'Shqiptarët e Malit të Zi'.

Retrieved items:
<item 1>
Darza
Darza (Albanian: "Darzë" , Montenegrin and ) is a village in the Ulcinj Municipality, southeastern Montenegro.

 It is a multi-ethnic settlement, inhabited by Montenegrins, Serbs and Albanians.

 According to the 2003 census, the total population was 119.
</item 1>

<item 2>
Albania
Albania is a unitary parliamentary constitutional republic with the capital in Tirana, the country's largest city and main economic and commercial centre,
followed by Durrës.

 The country's other major cities include Vlorë, Sarandë, Shkodër, Berat, Korçë, Gjirokastër and Fier.
</item 2>

<item 3>
Minority languages of Montenegro
European Charter for Regional or Minority Languages entered into force in Montenegro in June 2006, following the independence of Montenegro from the State Union
of Serbia and Montenegro on 3 June 2006.

 Constitution of Montenegro from 2007 states that Montenegrin language is official language of country, while Bosnian, Croatian, Serbian and Albanian are
languages in official use.

 The Constitution states that languages in official use are those of groups that form at least 1% of the population of Montenegro, as per 2003 population
census.

 The Law on National Minorities specifies that the percentage of members of national minorities in total population of the local government should be 15% in
order for their language and script to be introduced in official use.

 Media whose founder is Montenegro government are obliged to broadcast news, cultural, educational, sports and entertainment programs in minority languages.

 Minorities and their members have the right to education in their language in regular and vocational education.
</item 3>

<item 4>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 4>

<item 5>
Albanians
Albanians (Albanian: "Shqiptarët" ) are a nation and ethnic group, native to Albania, Kosovo and other countries who share a common culture, ancestry and speak
the Albanian language as a native tongue.

 Legally, the term is used to refer to the citizens of the Republic of Albania.

 Ethnic Albanians speak the Albanian language and more than half of ethnic Albanians live in Albania and Kosovo.

 A large Albanian population lives in Greece, Italy, the Republic of Macedonia, with smaller Albanian populations located in Serbia and Montenegro.
</item 5>

<item 6>
Albanians in the Republic of Macedonia
Albanians are the largest ethnic minority in the Republic of Macedonia.

 Of the 2,022,547 citizens of Macedonia, 509,083, or 25.2%, are Albanian according to the latest national census in 2002.

 The Albanian minority lives mostly in the north-western part of the country.

 The largest Albanian communities are in the municipalities of Tetovo (70.3% of the total population), Gostivar (66.7%), Debar (58.1%), Struga (56.8%), Kičevo
(54.5%), Kumanovo (25.8%) and Skopje (20.4%).
</item 6>

<item 7>
Albanian Coalition &quot;Perspective&quot;
The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
</item 7>

<item 8>
Graph query row:
organization: Albanian Coalition "Perspective"
kind: political party
relationship: LOCATED_IN
place: Montenegro
</item 8>

<item 9>
Albania
Albania ( , ; Albanian: "Shqipëri/Shqipëria" ; Gheg Albanian: "Shqipni/Shqipnia or Shqypni/Shqypnia" ), officially the Republic of Albania (Albanian: "Republika
e Shqipërisë" , ] ), is a country in Southern and Southeastern Europe.

 The country spans 28,748 km2 and had a total population of almost 3 million people as of 2016 .

 Albania is located in the southwestern part of the Balkan Peninsula, bordered by Montenegro to the northwest, Kosovo to the northeast, the Republic of
Macedonia to the east, and Greece to the south and southeast.

 The country has a coastline on the northern shore of the Mediterranean Sea, the Adriatic Sea to the west and the Ionian Sea to the southwest, forming the
Albanian Riviera.

 Albania is less than 72 km from Italy, across the Strait of Otranto which connects the Adriatic Sea to the Ionian Sea.
</item 9>

<item 10>
Albanians in Montenegro
Albanians in Montenegro (, Albanian: "Shqiptarët e Malit të Zi" ) constitute 4.91% of the county's total population.

 Albanians of Montenegro are Ghegs who mainly live in southeastern and eastern Montenegro, mainly in the following municipalities: Ulcinj (71% of population),
Plav (19%), Bar (6%), Podgorica (5%) and Rožaje (5%).
</item 10>
=== END ITEM id=cd03a1aed3e68f6d ===

=== ITEM id=7d2e51b5ec99452e ===
Gold sentences and facts:
Sentence 0: James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more than 70 productions to his credit.
  Fact 0.0: James David Sharman was born on 12 March 1945.
  Fact 0.1: James David Sharman is the son of boxing tent entrepreneur Jimmy Sharman.
  Fact 0.2: James David Sharman is an Australian director.
  Fact 0.3: James David Sharman is an Australian writer for film and stage.
  Fact 0.4: James David Sharman has more than 70 productions to his credit.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born on August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is an American director.
  Fact 1.2: Ava Marie DuVernay is an American screenwriter.
  Fact 1.3: Ava Marie DuVernay is an American film marketer.
  Fact 1.4: Ava Marie DuVernay is an American film distributor.

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
♯JusticeForFlint
Initiated by "Creed" director Ryan Coogler and "Selma" director Ava DuVernay, the free public event was held at the Whiting Auditorium in Flint, Michigan.

 Hosted by comedian Hannibal Buress, it featured singers Janelle Monáe and Ledisi, as well as actor-activists Jesse Williams and Jussie Smollett, amongst
others.

 Surprising star guest was "King of Motown" Stevie Wonder, who teamed up with Monáe for "Hell You Talkin' Bout", a protest song she had written for the people
of Flint.
</item 3>

<item 4>
Jim Sharman
James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more
than 70 productions to his credit.

 He is renowned in Australia for his work as a theatre director from the 1960s to the present, and is best known internationally as the director of the 1973
theatrical hit "The Rocky Horror Show", its film adaptation "The Rocky Horror Picture Show" (1975) and the film's follow-up "Shock Treatment" (1981).
</item 4>

<item 5>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 5>

<item 6>
The Assassin (2015 film)
The Assassin (; or: "The Assassin Niè Yǐnniáng") is a 2015 wuxia film directed by Taiwanese director Hou Hsiao-hsien.

 A Taiwan/China/Hong Kong co-production, it was an official selection in the main competition section at the 2015 Cannes Film Festival.

 At Cannes, Hou won the award for Best Director.

 It was released in China and Hong Kong on 27 August, and a day later in Taiwan on 28 August 2015.

 It was selected as the Taiwanese entry for the Best Foreign Language Film at the 88th Academy Awards but it was not nominated.
</item 6>

<item 7>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 7>

<item 8>
Middle of Nowhere (2012 film)
Middle of Nowhere is a 2012 independent feature film written and directed by Ava DuVernay and starring Emayatzy Corinealdi, David Oyelowo, Omari Hardwick and
Lorraine Toussaint.

 The film was the winner of the Directing Award for U.S. Dramatic Film at the 2012 Sundance Film Festival.
</item 8>

<item 9>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 9>

<item 10>
List of accolades received by Selma (film)
"Selma" is a 2014 American historical drama film directed by Ava DuVernay, and produced by Dede Gardner, Jeremy Kleiner, Christian Colson, and Oprah Winfrey.

 The screenplay was written by Paul Webb.

 The film follows the events leading up to and during the Selma to Montgomery marches, and the resulting establishment of the Voting Rights Act of 1965 which
prohibited racial discrimination in voting in the United States.

 The film stars David Oyelowo as civil rights activist Martin Luther King Jr., Tom Wilkinson as President Lyndon B. Johnson, and Tim Roth as Governor of Alabama
George Wallace.

 Carmen Ejogo, Dylan Baker, Wendell Pierce, Common, and Winfrey feature in supporting roles.
</item 10>
=== END ITEM id=7d2e51b5ec99452e ===

=== ITEM id=34c08d70345a7419 ===
Gold sentences and facts:
Sentence 0:  Levitt & Sons developed the Monmouth Heights section from 1965-1969.
  Fact 0.0: Levitt & Sons developed the Monmouth Heights section from 1965 to 1969.
Sentence 1:  The company is most famous for having built the town of Levittown, New York.
  Fact 1.0: Levitt & Sons is most famous for having built the town of Levittown, New York.

Retrieved items:
<item 1>
Monmouth Heights at Manalapan, New Jersey
Monmouth Heights at Manalapan is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 Levitt & Sons developed the Monmouth Heights section from 1965-1969.

 Residents of the 685 homes in the development are required to join the Monmouth Heights Community Association, which provides pools and other amenities.
</item 1>

<item 2>
Hanbys Corner, Delaware
Hanbys Corner is an unincorporated community in New Castle County, Delaware, United States.

 It is located at the intersection of Delaware Route 3 (Marsh Road) and Delaware Route 92 (Naamans Road), in Brandywine Hundred.

 The area is named for Richard G. Hanby, who first purchased the 125 acre parcel from the descendants of William Penn in 1753.

 His descendants included several notable figures in the local political scene including Samuel Winfield Hanby (1817-1892) who was elected as a State
Representative in 1874, Jacob Klose Hanby (1839-1932) who was Samuel's son and was elected State Representative in 1904, Robert Johnson Hanby (1834-1898), who
served in the 124th PA infantry during the Civil War and was elected State Senator in 1896, and Florence Wood Hanby (1870-1963), who was Robert's
daughter-in-law and was the first woman elected to the Delaware House of Representatives in 1924.

 In addition Wayne Hanby and James Hanby have both served as Justices of the Peace for New Castle County.
</item 2>

<item 3>
Province of New Jersey
The Province of New Jersey was one of the Middle Colonies of Colonial America and became the U.S. state of New Jersey in 1776.

 The province had originally been settled by Europeans as part of New Netherland, but came under English rule after the surrender of Fort Amsterdam in 1664,
becoming a proprietary colony.

 The English then renamed the province after the Isle of Jersey in the English Channel.

 The Dutch Republic reasserted control for a brief period in 1673–1674.

 After that it consisted of two political divisions, East Jersey and West Jersey, until they were united as a royal colony in 1702.

 The original boundaries of the province were slightly larger than the current state, extending into a part of the present state of New York, until the border
was finalized in 1773.
</item 3>

<item 4>
Levitt &amp; Sons
Levitt & Sons was a real estate development company founded by Abraham Levitt and later managed by his son William Levitt.

 The company is most famous for having built the town of Levittown, New York.

 The company's designs and building practices totally revolutionized the entire Home Building Industry and altered America's landscape with massive suburban
communities.
</item 4>

<item 5>
Toland, Pennsylvania
Toland is an unincorporated community in Cumberland County, Pennsylvania, United States.

 The very small village is located on Pine Grove Road, east of Mountain Creek Campground.

 Michaux State Forest, the site of Laurel Lake, Fuller Lake and Pine Grove Furnace State Park lie a few miles to the west near the intersection of Pine Grove
Road and PA route 233.

 Toland has a mailing address of Gardners Pennsylvania, because the size of the community doesn't warrant a post-office, nor an official incorporated name.

 What is now a sand pit operation was originally a clay bank mining operation that was the reason for the location of the village of Toland.

 Less than 50 people reside in the close-knit community, which was built for the clay bank company workers in the first quarter of the 20th century.

 The original community of Toland, Pennsylvania consisted of 11 duplex houses, built side by side along Pine Grove Road, with less than 0.17 of an acre of land
to each.

 Additional homes have been built since.

 When the community was originally built there was only a common well with a hand pump for all of the families to draw from.
</item 5>

<item 6>
Oakland Mills, New Jersey
Oakland Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The area is named for Oakland Mill, a mill along the Manalapan Brook operated by the Boo family.

 Some areas within the Manalapan Brook valley and surround forestland are preserved while the remainder of the area surrounding Oakland Mills has since been
developed into housing developments.
</item 6>

<item 7>
Monmouth Heights at Marlboro, New Jersey
Monmouth Heights at Marlboro is an unincorporated community located within Marlboro Township in Monmouth County, New Jersey, United States.

 The housing development was created in the 1960s by Levitt & Sons though some protests occurred by long-time residents of Marlboro Township at the time of its
creation.

 The community is located along New Jersey Route 79 across the highway from Marlboro High School.
</item 7>

<item 8>
HMS Monmouth
Seven ships of the Royal Navy have been named HMS "Monmouth".

 Monmouth was the name of a castle and is now the name of a town in Wales; the name also recognises James Scott, 1st Duke of Monmouth, the "Black Duke".

 The first HMS "Monmouth" was named after the town but whilst the second one was being used the Monmouth Rebellion took place and this caused some concern to
the crew.

 Since that time, all of "Monmouth"s have had their names painted in black and the name and the crest are covered because of the shame of the Duke of Monmouth's
execution.

 Today the officers of the latest HMS "Monmouth" carry black handkerchiefs and the ship is known as the Black Duke.

 However the town of Monmouth still gives the freedom of the town to the ship's crew and talks of the "enduring link" between the town and the ship.
</item 8>

<item 9>
Taylors Mills, New Jersey
Taylors Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The main road that runs through the area is named for the community, Taylors Mills Road.
</item 9>

<item 10>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
</item 10>
=== END ITEM id=34c08d70345a7419 ===

=== ITEM id=5d77947c08da46a0 ===
Gold sentences and facts:
Sentence 0: Operalia, The World Opera Competition is an annual international competition for young opera singers.
  Fact 0.0: Operalia, The World Opera Competition is an annual international competition.
  Fact 0.1: Operalia is a competition for young opera singers.
Sentence 1:  Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti, Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
  Fact 1.0: Operalia was founded in 1993.
  Fact 1.1: Operalia was founded by Plácido Domingo.
  Fact 1.2: Operalia has helped launch the careers of several important artists.
  Fact 1.3: Operalia has helped launch the career of Joseph Calleja.
  Fact 1.4: Operalia has helped launch the career of Giuseppe Filianoti.
  Fact 1.5: Operalia has helped launch the career of Rolando Villazón.
  Fact 1.6: Operalia has helped launch the career of José Cura.
  Fact 1.7: Operalia has helped launch the career of Joyce DiDonato.
  Fact 1.8: Operalia has helped launch the career of Elizabeth Futral.
  Fact 1.9: Operalia has helped launch the career of Inva Mula.
  Fact 1.10: Operalia has helped launch the career of Ana María Martínez.
  Fact 1.11: Operalia has helped launch the career of Sonya Yoncheva.
Sentence 2: Sonya Yoncheva (Bulgarian: Соня Йончева, born 25 December 1981) is a Bulgarian operatic soprano.
  Fact 2.0: Sonya Yoncheva was born on 25 December 1981.
  Fact 2.1: Sonya Yoncheva’s Bulgarian name is Соня Йончева.
  Fact 2.2: Sonya Yoncheva is a Bulgarian operatic soprano.

Retrieved items:
<item 1>
Andrea Carroll (soprano)
Andrea Carroll is an American soprano who has had an active international career in concerts and operas since 2012.

 A finalist in the 2015 Operalia, The World Opera Competition, she is particularly known for her performances with the Houston Grand Opera and the Vienna State
Opera.
</item 1>

<item 2>
Bogdan Volkov
Bogdan Volkov is a Russian operatic tenor.

 Born in Ukraine, one of the most young talented tenors of his generation.

 He is the winner of second prize at Plácido Domingo’s Operalia, The World Opera Competition in Guadalajara, Mexico in 2016 and first prize at the Paris Opera
Competition in 2015.
</item 2>

<item 3>
Guanqun Yu
Guanqun Yu (; born 1982) is a Chinese soprano who has sung in opera houses and concert halls internationally.

 In 2008 she won the Belvedere International Singing Competition and in 2012 she placed 2nd in the Operalia, The World Opera Competition.

 She is particularly known for portraying heroines in the operas of Wolfgang Amadeus Mozart, Giacomo Puccini, and Giuseppe Verdi.
</item 3>

<item 4>
Maria Katzarava
Maria Katzarava (born 1984) is an opera singer from Mexico who achieved international fame by winning first place in the Operalia competition in the opera and
zarzuela categories.

 Prior to this, she came to the attention of notable Mexican tenor Ramón Vargas, who helped her get a scholarship, and then won Mexico's Carlo Morelli national
singing competition in 2005, which earned her a debut at the Palacio de Bellas Artes.

 Since winning the Operalia competition, Katzarava has performed in Europe, where she currently lives, Asia and the Americas.

 She currently trains with soprano Mirella Freni.
</item 4>

<item 5>
Operalia, The World Opera Competition
Operalia, The World Opera Competition is an annual international competition for young opera singers.

 Founded in 1993 by Plácido Domingo, the competition has helped launch the careers of several important artists, such as Joseph Calleja, Giuseppe Filianoti,
Rolando Villazón, José Cura, Joyce DiDonato, Elizabeth Futral, Inva Mula, Ana María Martínez and Sonya Yoncheva.
</item 5>

<item 6>
Barbara Frittoli
Barbara Frittoli (born 19 April 1967) is an Italian operatic soprano who has sung leading roles in opera houses throughout Europe and in the United States.

 She was born in Milan and graduated from the Milan Conservatory.

 She made her Metropolitan Opera debut in 1995 as Micaela in "Carmen" and has gone on to sing in over 80 performances there including Donna Elvira in "Don
Giovanni", Fiordiligi in "Così fan tutte", Angelica in "Suor Angelica", Desdemona in "Otello", the title role in "Luisa Miller", Amelia in "Simon Boccanegra",
Vitellia in "La clemenza di Tito" and Alicia Ford in "Falstaff".
</item 6>

<item 7>
Rachel Willis-Sørensen
Rachel Willis-Sørensen (born 1984) is an American operatic soprano.

 She was a winner of the Metropolitan Opera National Council Auditions in 2010.

 In 2014, she won first prize, the Birgit Nilsson Prize, and the Zarzuela Prize of the Operalia competition.

 Other awards include first place in the 2009 Eleanor McCollum Competition for Young Singers in Houston and the 2010 Sara Tucker Grant from the Richard Tucker
Music Foundation.
</item 7>

<item 8>
Ao Li
Ao Li (born 8 February 1988) is a Chinese operatic bass-baritone and voice teacher who is particularly known for his performances at the San Francisco Opera.

 In 2013 he won first prize in the Operalia, The World Opera Competition and in 2014 he won the Metropolitan Opera National Council Auditions.
</item 8>

<item 9>
Sonya Yoncheva
Sonya Yoncheva (Bulgarian: Соня Йончева, born 25 December 1981) is a Bulgarian operatic soprano.
</item 9>

<item 10>
Joseph Calleja
Joseph Calleja, (born 22 January 1978 in Attard), is a Maltese tenor.

 He began singing at the age of 16, having been discovered by tenor Brian Cefai and continued his studies with Paul Asciak.

 At 19, he made his operatic debut as Macduff in Verdi's "Macbeth" at the Astra Theatre in Gozo and went on to become a prize winner at the Belvedere Hans Gabor
Competition the same year.

 In 1998, he won the Caruso Competition in Milan and was a prize winner in Plácido Domingo's Operalia International Opera Competition in 1999.
</item 10>
=== END ITEM id=5d77947c08da46a0 ===

=== ITEM id=e133ef020ff2b645 ===
Gold sentences and facts:
Sentence 0: Miltoniopsis, abbreviated Mltnps in horticultural trade, is a genus of orchids native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.
  Fact 0.0: Miltoniopsis is a genus of orchids.
  Fact 0.1: Miltoniopsis is abbreviated Mltnps in horticultural trade.
  Fact 0.2: Miltoniopsis is native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.
Sentence 1: Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few species extending north into temperate regions of Asia.
  Fact 1.0: Strobilanthes is a genus of about 350 species of flowering plants.
  Fact 1.1: Strobilanthes is in the family Acanthaceae.
  Fact 1.2: Strobilanthes is mostly native to tropical Asia and Madagascar.
  Fact 1.3: A few Strobilanthes species extend north into temperate regions of Asia.

Retrieved items:
<item 1>
Miltoniopsis
Miltoniopsis, abbreviated Mltnps in horticultural trade, is a genus of orchids native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.

 This genus comprises 5 species.

 Miltoniopsis's common name is Pansy Orchid.
</item 1>

<item 2>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 2>

<item 3>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 3>

<item 4>
Hemigraphis
Hemigraphis is a genus of plants in the family Acanthaceae, consisting of about 30 species native to tropical Asia.

 "Hemigraphis" is sometimes included in the genus "Strobilanthes"
</item 4>

<item 5>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 5>

<item 6>
Strobilanthes japonica
Strobilanthes japonica is a flowering herbaceous perennial plant from Asia, one of around 350 plants of the genus "Strobilanthes".

 The 20–50 cm ornamental plant is cultivated in Japan and China, and blooms in autumn with 1.5 cm purple to white funnel-shaped flowers.
</item 6>

<item 7>
Clinantheae
Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was described in 2000 by Alan Meerow "et al." as a result of a molecular phylogenetic study of the American Amaryllidoideae.

 This demonstrated that the Stenomesseae tribe, including the type genus "Stenomesson" was polyphyletic.

 Part of the tribe segregated with the Eucharideae and were submerged into it, while the other part formed a unique subclade.

 Since the type species of "Stenomesson" was not part of the second sublclade it was necessary to form a new name for the remaining species together with the
other genera that remained.

 This was "Clinanthus", the oldest name for these species, and consequently the tribe Clinantheae.
</item 7>

<item 8>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 8>

<item 9>
Nepenthes
Nepenthes ( ), also known as tropical pitcher plants, is a genus of carnivorous plants in the monotypic family Nepenthaceae.

 The genus comprises roughly 150 species, and numerous natural and many cultivated hybrids.

 They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two
species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).

 The greatest diversity occurs on Borneo, Sumatra, and the Philippines, with many endemic species.

 Many are plants of hot, humid, lowland areas, but the majority are tropical montane plants, receiving warm days but cool to cold, humid nights year round.

 A few are considered tropical alpine, with cool days and nights near freezing.

 The name "monkey cups" refers to the fact that monkeys have been observed drinking rainwater from these plants.
</item 9>

<item 10>
Miltoniopsis vexillaria
Miltoniopsis vexillaria ("the flag-like "Miltoniopsis"") is a species of epiphytic orchid in the genus Miltoniopsis.
</item 10>
=== END ITEM id=e133ef020ff2b645 ===

=== ITEM id=04ed7b7af5d81773 ===
Gold sentences and facts:
Sentence 0: Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.
  Fact 0.0: Nelson Algren was born March 28, 1909.
  Fact 0.1: Nelson Algren died May 9, 1981.
  Fact 0.2: Nelson Algren was an American writer.
Sentence 1: Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
  Fact 1.0: Nathanael West was born Nathan Weinstein.
  Fact 1.1: Nathanael West was born October 17, 1903.
  Fact 1.2: Nathanael West died December 22, 1940.
  Fact 1.3: Nathanael West was an American author.
  Fact 1.4: Nathanael West was an American screenwriter.
  Fact 1.5: Nathanael West was an American satirist.

Retrieved items:
<item 1>
Andrew Glaze
Andrew Glaze (April 21, 1920 – February 7, 2016) was an American poet, playwright and novelist.

 Much of Glaze's poetry reflects his coming of age in the South, and his eventual return there.

 He also lived and wrote in New York City for 31 years.

 In New York City he became part of a circle of poets that included Oscar Williams, Norman Rosten, John Ciardi and William Packard.
</item 1>

<item 2>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 2>

<item 3>
Nelson (band)
Nelson is an American rock band founded by singer/songwriters Matthew and Gunnar Nelson (twin sons of Ricky Nelson and Kristin Nelson).

 The band achieved success during the early 1990s with their double platinum debut album "After the Rain", which featured the number-one hit "(Can't Live
Without Your) Love and Affection".
</item 3>

<item 4>
Nelson Algren
Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.

 He may be best known for "The Man with the Golden Arm", a 1949 novel that won the National Book Award and was adapted as a 1955 film of the same name.
</item 4>

<item 5>
Nathanael West
Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
</item 5>

<item 6>
The Man with the Golden Arm (novel)
The Man with the Golden Arm is a novel by Nelson Algren, published by Doubleday in November 1949.

 One of the seminal novels of post-World War II American letters, "The Man with the Golden Arm" is widely considered Algren's greatest and most enduring work.

 It won the National Book Award in 1950.
</item 6>

<item 7>
Romeo Nelson
Iromeio "Romeo" Nelson (March 12, 1902 – May 17, 1974) was an American boogie woogie pianist whose recordings from 1929 are regarded as some of the finest, and
certainly the fastest, boogie woogie showpieces on record.
</item 7>

<item 8>
Chicago: City on the Make
Chicago: City on the Make is an essay by Nelson Algren published in 1951.

 Initially greeted with scorn by critics and newspaper editors in the city of its gaze (The "Chicago Daily News" famously called it a "Case for Ra(n)t
Control"), it is now widely regarded by scholars as the definitive prose portrait of the city of Chicago, although it has never rivaled the literary status of
Carl Sandburg's 1916 poem "Chicago."

 Algren leans heavily on the imagery and themes developed by Sandburg, to whom Algren dedicated the book.

 Curiously, he also quietly leans upon a poem about New York called "The City" by Ben Maddow, from whom Algren lifted powerful images of urban life.

 Subsequent portraits of Chicago, such as Studs Terkel's 1985 "Chicago", have likewise leaned heavily upon Algren's work.
</item 8>

<item 9>
A Walk on the Wild Side
A Walk on the Wild Side is a 1956 novel by Nelson Algren, most often quoted as the source for Algren's "three rules of life": "Never play cards with a man
called Doc.

 Never eat at a place called Mom's.

 Never sleep with a woman whose troubles are worse than your own."
</item 9>

<item 10>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 10>
=== END ITEM id=04ed7b7af5d81773 ===

=== ITEM id=fa30fc47795fab11 ===
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
Computer Magazine
Computer Magazine (and their website www.ComputerMagazine.com) is a popular magazine and online news site on computing and technology, offering current news and
reviews of popular and new business and consumer technologies, software, hardware, mobile computing, tablets, PCs, Macs, Windows, Linux, telecom, cellular,
wireless, data, cloud and science news on digital technologies and everything in the "tech-sphere and digi-verse", especially focused on information technology,
devices, software and services and related subjects, such as networking, servers, data centers and corporate data infrastructure technologies, and the Internet.

 Their online site, since 1997, is located at ComputerMagazine.com.

 "Computer Magazine" produces industry instructional and a popular ongoing webcast/podcast talk show and performs evaluations and reviews of IT industry
technology products, hardware, software and services with objective reporting widely respected as independent and objective, and trusted in the industry.
</item 1>

<item 2>
Popular Science
Popular Science (also known as PopSci) is an American bi-monthly magazine carrying popular science content, which refers to articles for the general reader on
science and technology subjects.

 "Popular Science" has won over 58 awards, including the American Society of Magazine Editors awards for its journalistic excellence in both 2003 (for General
Excellence) and 2004 (for Best Magazine Section).

 With roots beginning in 1872, "Popular Science" has been translated into over 30 languages and is distributed to at least 45 countries.
</item 2>

<item 3>
Popular science
Popular science (also pop-science or popsci) is interpretation of science intended for a general audience.

 While science journalism focuses on recent scientific developments, popular science is more broad-ranging.

 It may be written by professional science journalists or by scientists themselves.

 It is presented in many forms, including books, film and television documentaries, magazine articles, and web pages.
</item 3>

<item 4>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 4>

<item 5>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 5>

<item 6>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 6>

<item 7>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 7>

<item 8>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 8>

<item 9>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 9>

<item 10>
Computer Magazine
"Computer Magazine" is a free publication (in addition to their webcasts and other resources) sponsored by the nonprofit UTCP (United Technology and Computing
Professionals) organization, and as such charges no fees for the publication nor is influenced by advertising, so their reviews are relied on in the industry
and considered unbiased and thorough.

 "Computer Magazine" is one of the early large technology publications and resources available on the web still existent and thriving today and that has
remained independent.

 ComputerMagazine.com is a tech news and resources consolidator that publishes part of the site in a semi-time line/blogging format that is popular among their
wide following of subscriber and non-subscriber readers, allowing readers to respond and comment on various articles.

 Site contributors include many of the well known technology authors, experts and publication sources, content and articles are provided by major technology
syndicators and by external expert technology sources (such as "Computer World", "Information Week", "Network World", "Wired," "Time", etc.)
</item 10>
=== END ITEM id=fa30fc47795fab11 ===
