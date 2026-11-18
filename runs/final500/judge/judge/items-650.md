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

This file has 1 items.

=== ITEM id=f29de2fab6dd8106 ===
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
I Would Like to See You Again
I Would Like to See You Again is the 57th album by American country singer Johnny Cash, released on Columbia Records in 1978.

 The title track peaked at #12 on the singles chart, while "There Ain't No Good Chain Gang" reached #2; the album itself peaked at #23.

 The album features a pair of duets with Waylon Jennings, one of which was the "There Ain't No Good Chain Gang" single; it was one of Cash's first
collaborations with Jennings, and the two recorded songs together throughout the 1980s, including a separate album entitled "Heroes".

 Cash and Jennings would also work together as The Highwaymen with Willie Nelson and Kris Kristofferson.
</item 1>

<item 2>
The material covers country, country-rock, folk-rock and bluegrass artists and releases. Oak Ridge Boys’ 2009 *The Boys Are Back* marked a return to secular
country, led by a White Stripes cover. Richie Furay helped form Buffalo Springfield and Poco, later joining the Souther–Hillman–Furay Band; Poco’s debut
featured his “Pickin’ Up the Pieces.” Other subjects include Buffalo Springfield’s final album, the Lonesome Pine F
</item 2>

<item 3>
There Ain't No Good Chain Gang
"There Ain't No Good Chain Gang" is a song written by Hal Bynum and Dave Kirby, and recorded by American country music artists Johnny Cash and Waylon Jennings.

 It was released in May 1978 as the second single from the album "I Would Like to See You Again".

 The song reached #2 on the "Billboard" Hot Country Singles & Tracks chart.
</item 3>

<item 4>
Confederate Railroad
Confederate Railroad is an American country rock–Southern rock band founded in 1987 in Marietta, Georgia, by Danny Shirley (lead vocals, rhythm guitar), Michael
Lamb (lead guitar), Mark Dufresne (drums), Chris McDaniel (keyboards), Warren "Gates" Nichols (steel guitar) and Wayne Secrest (bass guitar).

 After serving as a backing band for outlaw country acts David Allan Coe and Johnny Paycheck, the band signed to a recording contract with Atlantic Records,
releasing their self-titled debut album that year.

 In the 1990s, they released four more albums for Atlantic.
</item 4>

<item 5>
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
</item 5>

<item 6>
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
</item 6>

<item 7>
The Closer You Get...
The Closer You Get... is the seventh studio album by country music band Alabama, released in 1983.

 All three singles from this album — "The Closer You Get", "Lady Down on Love" and "Dixieland Delight" — reached Number One on the Hot Country Songs charts in
1983.

 "She Put the Sad in All His Songs" was also recorded by Ronnie Dunn (who formed the duo Brooks & Dunn with Kix Brooks in 1991) and was released by him as a
single in 1983.
</item 7>

<item 8>
Lonesome Pine Fiddlers
In 1954, Charlie did a session, playing lead guitar, with the Stanley Brothers and also another one on RCA with the Fiddlers, although he was not otherwise
working with them at the time.

 He also worked briefly as a sideman with the Osborne Brothers, although he did not record with them.

 By 1958, Charlie (electric lead guitar) and his wife, Lee (electric bass), had rejoined Ezra and Curly Ray in the Lonesome Pine Fiddlers, who were
experimenting with a more modern sound and working a TV show in Huntington, West Virginia, in addition to daily radio in Pikeville.

 In his later years, Charlie was with the Stanley Brothers.

 Curly Ray also played with the Stanley Brothers at a different time as their fiddler.

 Curly Ray was one of the best fiddlers in Bluegrass.

 This most talented family of musicians were the best, surpassed by none.
</item 8>

<item 9>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 9>

<item 10>
The Highwaymen (country supergroup)
The Highwaymen was a country music supergroup composed of four of the genre's biggest artists well known for their pioneering influence on the outlaw country
subgenre: Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.

 Active as a group between 1985 and 1995, these four artists recorded three major label albums as The Highwaymen: two on Columbia Records and one for Liberty
Records.

 Their Columbia works produced three chart singles, including the number one "Highwayman" in 1985.
</item 10>
=== END ITEM id=f29de2fab6dd8106 ===
