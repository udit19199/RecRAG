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

This file has 6 items.

=== ITEM id=5af4bedd7d6e7e21 ===
Gold sentences and facts:
Sentence 0: Other applications are short - circuit - proof extra-low voltage transformers for toys or doorbell installations .
  Fact 0.0: Other applications are short-circuit-proof extra-low voltage transformers for toys.
  Fact 0.1: Other applications are short-circuit-proof extra-low voltage transformers for doorbell installations.

Retrieved items:
<item 1>
Transformer types
Loudspeaker transformer ( edit ) Loudspeaker transformer in old radio In the same way that transformers create high voltage power transmission circuits that
minimize transmission losses , loudspeaker transformers can power many individual loudspeakers from a single audio circuit operated at higher than normal
loudspeaker voltages . This application is common in public address applications . Such circuits are commonly referred to as constant - voltage speaker systems
. Such systems are also known by the nominal voltage of the loudspeaker line , such as 25 - , 70 - and 100 - volt speaker systems ( the voltage corresponding to
the power rating of a speaker or amplifier ) . A transformer steps up the output of the system 's amplifer to the distribution voltage . At the distant
loudspeaker locations , a step - down transformer matches the speaker to the rated voltage of the line , so the speaker produces rated nominal output when the
line is at nominal voltage . Loudspeaker transformers commonly have multiple primary taps to adjust the volume at each speaker in steps .
</item 1>

<item 2>
Transformer types
It acts then both as a voltage transformer and as a magnetic ballast . Other applications are short - circuit - proof extra-low voltage transformers for toys or
doorbell installations . Resonant transformer ( edit ) Further information : Resonant inductive coupling Further information : Tesla coil § Resonant transformer
A resonant transformer is a transformer in which one or both windings has a capacitor across it and functions as a tuned circuit . Used at radio frequencies ,
resonant transformers can function as high Q factor bandpass filters . The transformer windings have either air or ferrite cores and the bandwidth can be
adjusted by varying the coupling ( mutual inductance ) . One common form is the IF ( intermediate frequency ) transformer , used in superheterodyne radio
receivers . They are also used in radio transmitters . Resonant transformers are also used in electronic ballasts for gas discharge lamps , and high voltage
power supplies . They are also used in some types of switching power supplies . Here the short - circuit inductance value is an important parameter that
determines the resonance frequency of the resonant transformer .
</item 2>

<item 3>
Transformer types
1 Constant voltage transformer 1.12 Ferrite core 1.12. 1 Planar transformer 1.13 Oil cooled transformer 1.14 Cast resin transformer 1.15 Isolating transformer 2
Instrument transformer 2.1 Current transformer 2.2 Voltage transformer or potential transformer 2.3 Combined instrument transformer 3 Pulse transformer 4 RF
transformer 4.1 Air - core transformer 4.2 Ferrite - core transformer 4.3 Transmission - line transformer 4.4 Balun 5 IF transformer 6 Audio transformer 6.1
Loudspeaker transformer 6.2 Output transformer 6.3 Small signal transformer 6.4 Interstage and coupling transformers 7 Other types 7.1 Transactor 7.2 Hedgehog
7.3 Variometer and variocoupler 7.4 Rotary transformer 7.5 Variable differential transformer 7.6 Resolver and synchro 8 See also 9 References Power transformers
( edit ) Laminated core ( edit ) Laminated core transformer This is the most common type of transformer , widely used in electric power transmission and
appliances to convert mains voltage to low voltage to power electronic devices .
</item 3>

<item 4>
Transformer types
Audio transformers were originally designed to connect different telephone systems to one another while keeping their respective power supplies isolated , and
are still commonly used to interconnect professional audio systems or system components , to eliminate buzz and hum . Such transformers typically have a 1 : 1
ratio between the primary and the secondary . These can also be used for splitting signals , balancing unbalanced signals , or feeding a balanced signal to
unbalanced equipment . Transformers are also used in DI boxes to convert high - impedance instrument signals ( e.g. , bass guitar ) to low impedance signals to
enable them to connect to a microphone input on the mixing console . A particularly critical component is the output transformer of a valve amplifier . Valve
circuits for quality reproduction have long been produced with no other ( inter-stage ) audio transformers , but an output transformer is needed to couple the
relatively high impedance ( up to a few hundred ohms depending upon configuration ) of the output valve ( s ) to the low impedance of a loudspeaker . ( The
valves can deliver a low current at a high voltage ; the speakers require high current at low voltage . ) Most solid - state power amplifiers need no output
transformer at all .
</item 4>

<item 5>
Graph query row:
transformer: extra-low voltage transformer
relationship: RELATED_TO
connected_entity: doorbell installations
fact: None
context: None
</item 5>

<item 6>
Transformer types
This is a common method used in portable current measuring instruments but permanent installations use more economical types of current transformer . Voltage
transformer or potential transformer ( edit ) Voltage transformers ( VT ) , also called potential transformers ( PT ) , are a parallel connected type of
instrument transformer , used for metering and protection in high - voltage circuits or phasor phase shift isolation . They are designed to present negligible
load to the supply being measured and to have an accurate voltage ratio to enable accurate metering . A potential transformer may have several secondary
windings on the same core as a primary winding , for use in different metering or protection circuits . The primary may be connected phase to ground or phase to
phase . The secondary is usually grounded on one terminal . There are three primary types of voltage transformers ( VT ) : electromagnetic , capacitor , and
optical . The electromagnetic voltage transformer is a wire - wound transformer . The capacitor voltage transformer uses a capacitance potential divider and is
used at higher voltages due to a lower cost than an electromagnetic VT . An optical voltage transformer exploits the electrical properties of optical materials
. Measurement of high voltages is possible by the potential transformers .
</item 6>

<item 7>
Transformer types
Applications : Intermediate frequency ( IF ) transformer in superheterodyne radio receiver Tank transformers in radio transmitters Tesla coil CCFL inverter
Oudin coil ( or Oudin resonator ; named after its inventor Paul Oudin ) D'Arsonval apparatus Ignition coil or induction coil used in the ignition system of a
petrol engine Electrical breakdown and insulation testing of high voltage equipment and cables . In the latter case , the transformer 's secondary is resonated
with the cable 's capacitance . Constant voltage transformer ( edit ) See also : Voltage regulator § Constant - voltage transformer By arranging particular
magnetic properties of a transformer core , and installing a ferro - resonant tank circuit ( a capacitor and an additional winding ) , a transformer can be
arranged to automatically keep the secondary winding voltage relatively constant for varying primary supply without additional circuitry or manual adjustment .
Ferro - resonant transformers run hotter than standard power transformers , because regulating action depends on core saturation , which reduces efficiency .
The output waveform is heavily distorted unless careful measures are taken to prevent this . Saturating transformers provide a simple rugged method to stabilize
an AC power supply .
</item 7>

<item 8>
Transformer types
Measurement of high voltages is possible by the potential transformers . Combined instrument transformer ( edit ) Combined instrument transformer A combined
instrument transformer encloses a current transformer and a voltage transformer in the same transformer . There are two main combined current and voltage
transformer designs : oil - paper insulated and SF insulated . One advantage of applying this solution is reduced substation footprint , due to reduced number
of transformers in a bay , supporting structures and connections as well as lower costs for civil works , transportation and installation . Pulse transformer (
edit ) Bothhand TS6121A pulse transformer A pulse transformer is a transformer that is optimised for transmitting rectangular electrical pulses ( that is ,
pulses with fast rise and fall times and a relatively constant amplitude ) . Small versions called signal types are used in digital logic and telecommunications
circuits , often for matching logic drivers to transmission lines . Medium - sized power versions are used in power - control circuits such as camera flash
controllers . Larger power versions are used in the electrical power distribution industry to interface low - voltage control circuitry to the high - voltage
gates of power semiconductors .
</item 8>

<item 9>
Graph query row:
transformer: extra-low voltage transformer
relationship: RELATED_TO
connected_entity: toys
fact: None
context: None
</item 9>

<item 10>
Transformer types
The epoxy protects the windings from dust and corrosive atmospheres . However , because the molds for casting the coils are only available in fixed sizes , the
design of the transformers is less flexible , which may make them more costly if customized features ( voltage , turns ratio , taps ) are required . Isolating
transformer ( edit ) An isolation transformer links two circuits magnetically , but provides no metallic conductive path between the circuits . An example
application would be in the power supply for medical equipment , when it is necessary to prevent any leakage from the AC power system into devices connected to
a patient . Special purpose isolation transformers may include shielding to prevent coupling of electromagnetic noise between circuits , or may have reinforced
insulation to withstand thousands of volts of potential difference between primary and secondary circuits . Instrument transformer ( edit ) Main article :
Instrument transformer Instrument transformers Instrument transformers are typically used to operate instruments from high voltage lines or high current
circuits , safely isolating measurement and control circuitry from the high voltages or currents . The primary winding of the transformer is connected to the
high voltage or high current circuit , and the meter or relay is connected to the secondary circuit .
</item 10>
=== END ITEM id=5af4bedd7d6e7e21 ===

=== ITEM id=0a2375f891474dcd ===
Gold sentences and facts:
Sentence 0: `` Where Have All the Flowers Gone ? '' is a modern folk - style song . The melody and the first three verses were written by Pete Seeger in 1955 and published in Sing Out ! magazine . Additional verses were added in May 1960 by Joe Hickerson , who turned it into a circular song . Its rhetorical `` where ? '' and meditation on death place the song in the ubi sunt tradition . In 2010 , the New Statesman listed it as one of the `` Top 20 Political Songs '' .
  Fact 0.0: Where Have All the Flowers Gone? is a modern folk-style song.
  Fact 0.1: The melody of Where Have All the Flowers Gone? was written by Pete Seeger in 1955.
  Fact 0.2: The first three verses of Where Have All the Flowers Gone? were written by Pete Seeger in 1955.
  Fact 0.3: The melody of Where Have All the Flowers Gone? was published in Sing Out! magazine.
  Fact 0.4: The first three verses of Where Have All the Flowers Gone? were published in Sing Out! magazine.
  Fact 0.5: Additional verses of Where Have All the Flowers Gone? were added in May 1960 by Joe Hickerson.
  Fact 0.6: Joe Hickerson turned Where Have All the Flowers Gone? into a circular song.
  Fact 0.7: The rhetorical 'where?' and meditation on death of Where Have All the Flowers Gone? place the song in the ubi sunt tradition.
  Fact 0.8: In 2010, the New Statesman listed Where Have All the Flowers Gone? as one of the 'Top 20 Political Songs'.

Retrieved items:
<item 1>
Where Have All the Flowers Gone?
Versions ( edit ) The Kingston Trio recorded the song in 1961 . Believing it to be a traditional song , they claimed authorship , although upon notice from
Seeger they had their name removed and credited Seeger and Hickerson . Seeger acknowledged their success with this song . Their single , with `` O Ken Karanga
'' as the A-side and the hit `` Where Have All The Flowers Gone ? '' the B - side , reached # 21 in the 1962 Billboard Hot 100 chart and # 4 on the Easy
Listening chart . Peter , Paul and Mary included the song on their eponymous debut album ( which spent five weeks as the # 1 album in the country ) in 1962 .
Marlene Dietrich performed the song in English , French , and German . The song was first performed in French ( as `` Qui peut dire où vont les fleurs ? '' ) by
Marlene in 1962 at a UNICEF concert . She also recorded the song in English and in German , the latter titled `` Sag ' mir , wo die Blumen sind '' , with lyrics
translated by Max Colpet .
</item 1>

<item 2>
Where Have All the Flowers Gone?
Version Title Artist Basque Loreak non dira ? Lou Topet , Harkaitz Cano Belarusian Дзе кветкі ўсе ? Bar Akaryna Bengali কোথায় গেল তারা ? ( Kothay Gelo Tara )
Kabir Suman Catalan Què se n'ha fet d'aquelles flors ? Roslyn Smith , La Marta ( Club Super3 ) Chinese 花 兒 怎麼 不見 了 ? Poon Sow Keng ( 潘秀瓊 ) Croatian Kamo je
cvijeće otišlo ? Monia Verardi Czech Řekni , kde ty kytky jsou Judita Čeřovská , Marie Rottrová , Marta Kubišová Danish Where Have All the Flowers Gone Savage
Rose Dutch Zeg me waar de bloemen zijn Jaap Fischer English Where have all the flowers gone ?
</item 2>

<item 3>
Where Have All the Flowers Gone?
Sing Out ! . 53 ( 2 ) : 76 . ^ Jump up to : Joe Hickerson . `` Where Have All the Flowers Gone ? '' . presentation to SEM ( Society for Ethnomusicology ) , 50th
Annual Meeting in Atlanta ( quoted in thread ) . Mudcat.org . Retrieved October 20 , 2009 . ^ Jump up to : Smith , Ian K ( March 25 , 2010 ) . `` Top 20
Political Songs : Where Have All the Flowers Gone '' . New Statesman . Retrieved March 25 , 2010 . Jump up ^ A Folk Legend 's Fertile Ground , Oberlin Alumni
Magazine , summer 2014 Jump up ^ Notes from : Where Have All the Flowers Gone - The Songs of Pete Seeger ^ Jump up to : Dunaway , David King ( 2008 ) . How Can
I Keep From Singing ? The Ballad of Pete Seeger , pp. 228 - 30 . Random House , Inc . ISBN 0 - 345 - 50608 - 1 .
</item 3>

<item 4>
Where Have All the Flowers Gone?
ISBN 0 - 345 - 50608 - 1 . Jump up ^ Pete Seeger - Little Boxes / Where Have All the Flowers Gone . Discogs.com . Jump up ^ Pete Seeger - The Bitter and the
Sweet . Discogs.com . Jump up ^ Where Have All the Flowers Gone ? Second Hand Songs . Jump up ^ The Kingston Trio time line . The single was released on
December 18 , 1961 by the group on Capitol Records as a 45 single , 4671 . Source : Liner notes for The Kingston Trio : The Capital Years ( Capitol Records
CD7243 8 28498 2 7 ) . Jump up ^ Gilliland , John ( 1969 ) . `` Show 18 - Blowin ' in the Wind : Pop discovers folk music . ( Part 1 ) '' ( audio ) . Pop
Chronicles . University of North Texas Libraries . Track 5 . Jump up ^ Whitburn , Joel ( 2002 ) . Top Adult Contemporary : 1961 - 2001 . Record Research . p.
137 .
</item 4>

<item 5>
Where Have All the Flowers Gone?
Roy Orbison recorded a version of the song which appears on the album The Connoisseur 's Orbison . Eddy Arnold and The Needmore Creek Singers recorded the song
on October 9 , 1963 , and released it on the Folk Song Book album released in January 1964 . Vera Lynn recorded the song as the eleventh cut on her 1964 album
`` Among My Souvenirs '' . The Brothers Four recorded the song on their 1964 LP `` More Big Folk Hits '' , Columbia Records , CL - 2213 . The Four Seasons
recorded the song on their 1964 Philips album Born to Wander , PHM 200 129 . Lars Lönndahl recorded the song in 1964 with Swedish lyrics Inga blommor finns det
mer , translated in 1962 by Beppe Wolgers . Joan Baez included the German version ( Sagt Mir wo die Blumen sind ) on her 1965 album Farewell Angelina .
</item 5>

<item 6>
Where Have All the Flowers Gone?
Johnny Rivers had a 1965 top 40 hit in the US with a folk rock version , reaching # 26 on the Billboard Hot 100 and # 9 in Canada . Grady Martin released an
instrumental version in 1965 on his Instrumentally Yours album . Harry Belafonte has made one recording of it at a benefit concert in Stockholm , Sweden , 1966
on the album BEL - 1 . Lester Flatt and Earl Scruggs included the song on their 1968 album Changin ' Times . Walter Jackson recorded a R & B version in Chicago
for Okeh records in 1967 . Jazz guitarist Wes Montgomery recorded an instrumental version in his trademark style on the 1968 LP Road Song In 1969 , Rufus Harley
recorded a jazz instrumental version on his trademark bagpipes , but the track was never commercially released until it was included in his posthumously
released limited edition collection Courage - The Atlantic Recordings in 2006 . The Landsmen released the song as a 45 single on Arvee . Bill Anderson recorded
the song on his eponymous 1971 album .
</item 6>

<item 7>
Where Have All the Flowers Gone?
The 1964 release of the song as a Columbia Records 45 single , 13 - 33088 , by Pete Seeger was inducted into the Grammy Hall of Fame in 2002 in the Folk
category . Contents ( hide ) 1 Composition 2 Versions 3 Grammy Hall of Fame 4 See also 5 References 6 Bibliography Composition ( edit ) Seeger found inspiration
for the song in October 1955 while he was on a plane bound for a concert at Oberlin College , one of the few venues which would hire him during the McCarthy era
. Leafing through his notebook he saw the passage , `` Where are the flowers , the girls have plucked them . Where are the girls , they 've all taken husbands .
Where are the men , they 're all in the army . '' These lines were taken from the traditional Cossack folk song `` Koloda - Duda '' , referenced in the Mikhail
Sholokhov novel And Quiet Flows the Don ( 1934 ) , which Seeger had read `` at least a year or two before '' . Seeger created a song which was subsequently
published in Sing Out in 1962 .
</item 7>

<item 8>
Where Have All the Flowers Gone?
Bill Anderson recorded the song on his eponymous 1971 album . American R&B band Earth , Wind & Fire covered the song on the 1972 album Last Days and Time .
Richie Havens recorded the song in 1972 . Yellow Magic Orchestra recorded the song in 1980 and released it in the 1999 compilation album `` YMO GO HOME ! ''
Hannes Wader recorded a German version as the final track for his 1982 album Daß nichts bleibt wie es war . City recorded a German version in their 1983 album
Unter der Haut . Bernie Sanders covered the song on his 1987 album We Shall Overcome . A Russian version of the song was created in 1998 by Oleg Nesterov , a
lead singer of Megapolis , and later performed in duet with Masha Makarova ( Masha I Medvedi ) in a music video . Olivia Newton - John recorded the song on her
2004 album Indigo : Women of Song . Country singer Dolly Parton has also recorded a rendition of the song , on her 2005 album Those Were the Days .
</item 8>

<item 9>
Where Have All the Flowers Gone?
Oğuz Tarihmen Ukrainian Де всі квіти , розкажи De vsi kvity , rozkazhy Maria Burmaka Ukrainian Квіти де ? Kvity De ? Yana Zavarzina Grammy Hall of Fame ( edit )
Pete Seeger 's recording of his composition was inducted into the Grammy Hall of Fame , which is a special Grammy award established in 1973 to honor recordings
that are at least 25 years old and that have `` qualitative or historical significance . '' Pete Seeger : Grammy Hall of Fame Awards Year recorded Title Genre
Label Year inducted 1964 `` Where Have All the Flowers Gone ? '' Folk ( single ) Columbia 2002 See also ( edit ) List of anti-war songs References ( edit ) Jump
up ^ Seeger , Pete . `` Where Have All the Flowers Gone '' . Sing Out ! . 11 ( 5 ) . Jump up ^ Hickerson , Joe ( 2009 -- 2010 ) . `` The Songfinder '' . Sing
Out ! . 53 ( 2 ) : 76 .
</item 9>

<item 10>
Where Have All the Flowers Gone?
Where Have All the Flowers Gone ? - Wikipedia Where Have All the Flowers Gone ? Jump to : navigation , search For the 2008 documentary , see Where Have All the
Flowers Gone ? ( film ) . 1960 release on The Rainbow Quest LP , Folkways , FA 2454 , by Pete Seeger Columbia Hall of Fame 45 rpm single release as 13 - 33088
featuring the November 1962 version . August 1965 . `` Where Have All The Flowers Gone ? '' Song Language English Composer ( s ) Pete Seeger Lyricist ( s ) Pete
Seeger and Joe Hickerson `` Where Have All the Flowers Gone ? '' is a modern folk - style song . The melody and the first three verses were written by Pete
Seeger in 1955 and published in Sing Out ! magazine . Additional verses were added in May 1960 by Joe Hickerson , who turned it into a circular song . Its
rhetorical `` where ? '' and meditation on death place the song in the ubi sunt tradition . In 2010 , the New Statesman listed it as one of the `` Top 20
Political Songs '' .
</item 10>
=== END ITEM id=0a2375f891474dcd ===

=== ITEM id=2c60a00d99e01305 ===
Gold sentences and facts:
Sentence 0: Loyalists were American colonists who remained loyal to the British Crown during the American Revolutionary War . At the time they were often called Tories , Royalists , or King 's Men . They were opposed by the Patriots , those who supported the revolution and called them `` persons inimical to the liberties of America '' . Prominent Loyalists repeatedly assured the British government that many thousands of loyalists would spring to arms and fight for the crown . The British government acted in expectation of that , especially in the southern campaigns in 1780 - 81 . In practice , the number of loyalists in military service was far lower than expected . Across the colonies , Patriots watched suspected Loyalists very closely , and would not tolerate any organized Loyalist opposition . Many outspoken or militarily active loyalists were forced to flee , especially to their stronghold of New York City . William Franklin , the royal governor of New Jersey and son of Patriot leader Benjamin Franklin , became the leader of the Loyalists after his release from a Patriot prison in 1778 . He worked to build Loyalist military units to fight in the war , but the number of volunteers was much fewer than London expected .
  Fact 0.0: Loyalists were American colonists who remained loyal to the British Crown during the American Revolutionary War.
  Fact 0.1: Loyalists were called Tories, Royalists, or King's Men at the time.
  Fact 0.2: Loyalists were opposed by the Patriots.
  Fact 0.3: The Patriots were those who supported the revolution.
  Fact 0.4: The Patriots called the Loyalists "persons inimical to the liberties of America."
  Fact 0.5: Prominent Loyalists repeatedly assured the British government that many thousands of loyalists would spring to arms and fight for the Crown.
  Fact 0.6: The British government acted in expectation of that, especially in the southern campaigns in 1780-81.
  Fact 0.7: In practice, the number of loyalists in military service was far lower than expected.
  Fact 0.8: Patriots watched suspected Loyalists very closely across the colonies.
  Fact 0.9: Patriots would not tolerate any organized Loyalist opposition.
  Fact 0.10: Many outspoken or militarily active loyalists were forced to flee.
  Fact 0.11: Loyalists fled especially to their stronghold of New York City.
  Fact 0.12: William Franklin was the royal governor of New Jersey.
  Fact 0.13: William Franklin was the son of Patriot leader Benjamin Franklin.
  Fact 0.14: William Franklin became the leader of the Loyalists after his release from a Patriot prison in 1778.
  Fact 0.15: William Franklin worked to build Loyalist military units to fight in the war.
  Fact 0.16: The number of volunteers for William Franklin's Loyalist military units was much fewer than London expected.

Retrieved items:
<item 1>
Loyalist (American Revolution)
Benjamin Franklin to Baron Francis Maseres , June 26 , 1785 ( Opinion of Benjamin Franklin on persons who called themselves `` Loyalists '' , whom he judged
better called `` Royalists '' ) Bibliography of the Loyalist Participation in the American Revolution compiled by the United States Army Center of Military
History `` Black Loyalists : Our History , Our People '' Haldimand Collection The main source for historians in the study of the settlement of the American
Loyalists in Canada . More than 20,000 letters and documents , now fully indexed , and free on the Web . James Chalmers and `` Plain Truth '' ( A Loyalist
Answers Thomas Paine ) The Loyalist Declaration of Independence published in The Royal Gazette , ( New York ) on November 17 , 1781 ( Transcription provided by
Bruce Wallace and posted on The On - Line Institute for Advanced Loyalist Studies . )
</item 1>

<item 2>
Loyalist (American Revolution)
They were alienated when the Patriots resorted to violence , such as burning houses and tarring and feathering . They wanted to take a middle - of - the road
position and were angry when forced by the Patriots to declare their opposition . They had a long - standing sentimental attachment to Britain ( often with
business and family links ) . They were procrastinators who realized that independence was bound to come someday , but wanted to postpone the moment . They were
cautious and afraid that chaos and mob rule would result . Some were pessimists who lacked the confidence in the future displayed by the Patriots . Others
recalled the dreadful experiences of many Jacobite rebels after the failure of the last Jacobite rebellion as recently as 1745 who often lost their lands when
the Hanoverian government won . Other motives of the Loyalists included : They felt a need for order and believed that Parliament was the legitimate authority .
In New York , powerful families had assembled colony - wide coalitions of supporters , Men long associated with the De Lancey faction went along when its
leadership decided to support the crown .
</item 2>

<item 3>
Loyalist (American Revolution)
Approximately half the colonists of European ancestry tried to avoid involvement in the struggle -- some of them deliberate pacifists , others recent immigrants
, and many more simple apolitical folk . The patriots received active support from perhaps 40 to 45 percent of the white populace , and at most no more than a
bare majority . A jury finding from Kentucky County , Virginia in July 1780 , confiscating lands of two men adjudged to be British subject . Daniel Boone was
listed as a member of the jury . Before Calhoon 's work , estimates of the Loyalist share of the population were somewhat higher , at about one - third , but
these estimates are now rejected as too high by most scholars . In 1968 historian Paul H. Smith estimated there were about 400,000 Loyalists , or 16 % of the
white population of 2.25 million in 1780 .
</item 3>

<item 4>
Loyalist (American Revolution)
1 A 9.2 B 9.3 C 9.4 D 9.5 E 9.6 F 9.7 G 9.8 H 9.9 I 9.10 J 9.11 K 9.12 L 9.13 M 9.14 N 9.15 O 9.16 P 9.17 R 9.18 S 9.19 T 9.20 V 9.21 W 9.22 X , Y , Z 10 See
also 11 References 12 Further reading 12.1 Compiled volumes of biographical sketches 12.2 Studies of individual Loyalists 12.3 Primary sources and guides to
manuscripts and the literature 13 External links 13.1 Videos Background ( edit ) The American Revolution was a civil war based on who would rule in the Thirteen
Colonies . Families were often divided as war forced colonists to choose sides in a conflict that remained for many years uncertain . Colonists , especially
recent arrivals , often felt themselves to be both American and British , subjects of the King , still owing a loyalty to the mother country .
</item 4>

<item 5>
Loyalist (American Revolution)
Historian Robert Middlekauff summarized scholarly research on the nature of Loyalist support as follows : The largest number of loyalists were found in the
middle colonies : many tenant farmers of New York supported the king , for example , as did many of the Dutch in the colony and in New Jersey . The Germans in
Pennsylvania tried to stay out of the Revolution , just as many Quakers did , and when that failed , clung to the familiar connection rather than embrace the
new . Highland Scots in the Carolinas , a fair number of Anglican clergy and their parishioners in Connecticut and New York , a few Presbyterians in the
southern colonies , and a large number of the Iroquois Indians stayed loyal to the king . Johnson Hall , seat of Sir John Johnson in the Mohawk Valley New York
City and Long Island were the British military and political base of operations in North America from 1776 to 1783 and had a large concentration of Loyalists ,
many of whom were refugees from other states . According to Calhoon , Loyalists tended to be older and wealthier , but there were also many Loyalists of humble
means .
</item 5>

<item 6>
Loyalist (American Revolution)
Chesterton Winston Churchill Enoch Powell George Grant Related topics ( show ) Conservatism Distributists Miguelism Vendéens Chouans Legitimism Carlism
Sanfedismo Viva Maria Cristeros Reactionary High Tory Red Tory Loyalist Royalist Veronese Easter Tory corporatism Tory socialism Pink Tory Ultra-Tories
Britannia offers solace and a promise of compensation for her exiled American - born Loyalists . ( Reception of the American Loyalists by Great Britain in the
Year 1783 . Engraving by H. Moses after Benjamin West . ) Loyalists were American colonists who remained loyal to the British Crown during the American
Revolutionary War . At the time they were often called Tories , Royalists , or King 's Men . They were opposed by the Patriots , those who supported the
revolution and called them `` persons inimical to the liberties of America '' . Prominent Loyalists repeatedly assured the British government that many
thousands of loyalists would spring to arms and fight for the crown . The British government acted in expectation of that , especially in the southern campaigns
in 1780 - 81 .
</item 6>

<item 7>
Loyalist (American Revolution)
1778 ) , led a band of Tories , commissioned by Lord Dunmore , who upon capture was executed John Randolph ( 1727 -- 1784 ) , King 's Attorney for Virginia Anna
Rawle , a women loyalist who used letter - diaries to spy and transfer information about Rebels in PA to loyalist in New York James Rivington 1734 - 1802
Loyalist editor of Rivington Gazette and a Patriot Spy Colonel Beverley Robinson ( 1723 -- 1792 ) , of New York , Loyal American Regiment Lieutenant - Colonel
Robert Rogers ( 1731 -- 1795 ) , commander of Rogers ' Rangers / Queen 's Rangers to 1777 ( now The Queen 's York Rangers ( 1st American Regiment ) ( RCAC ) ) ,
innovator of ranging tactics Peggy Shippen ( 1760 -- 1804 ) , Philadelphia socialite and second wife of Benedict Arnold John Graves Simcoe ( 1752 -- 1806 ) ,
commander of Queen 's Rangers from 1777 ( now The Queen 's York Rangers ( 1st American Regiment ) ( RCAC ) ,
</item 7>

<item 8>
Loyalist (American Revolution)
They felt themselves to be weak or threatened within American society and in need of an outside defender such as the British Crown and Parliament . They had
been promised freedom from slavery by the British . They felt that being a part of the British Empire was crucial in terms of commerce and their business
operations . Loyalism and military operations ( edit ) See also : Loyalists fighting in the American Revolution In the opening months of the Revolutionary War ,
the Patriots laid siege to Boston , where most of the British forces were stationed . Elsewhere there were few British troops and the Patriots seized control of
all levels of government , as well as supplies of arms and gunpowder . These actions were not without resistance . Especially in New York , New Jersey , and
parts of North and South Carolina , there was considerable ambivalence about the Patriot cause . Vocal Loyalists , often with the encouragement and assistance
of royal governors , recruited people to their side . In the South Carolina backcountry Loyalist recruitment oustripped that of Patriots .
</item 8>

<item 9>
Loyalist (American Revolution)
Many , like Maryland lawyer Daniel Dulaney , opposed taxation without representation , but would not break their oath to the King or take up arms against him .
In one of his many pamphlets , Dulaney wrote : `` There may be a time when redress may not be obtained . Till then , I shall recommend a legal , orderly , and
prudent resentment '' . Most hoped for a peaceful reconciliation , and were forced to choose sides by the Patriots who took control nearly everywhere in 1775 -
76 . Likely the earliest formal meeting , and use of the term `` royalist '' and the source of the United Empire Loyalist ( `` UEL '' ) acronym , took place in
Boston on October 28 , 1775 . At the meeting , held at the suggestion of General Thomas Gage , these Loyalists formed a society called `` The Loyalist
Associators despite the Unity of the Empire '' .
</item 9>

<item 10>
Loyalist (American Revolution)
George Washington 's Mount Vernon . Jump up ^ `` The Loyalists '' . Jump up ^ Calhoun ( 1973 ) Jump up ^ Georgia Encyclopædia . Jump up ^ Robert M. Calhoon , in
' A companion to the American Revolution ' , ( 2000 ) ; p 235 . Jump up ^ John Adams has sometimes been cited as having claimed , in an 1813 letter , that one -
third of Americans supported the revolution and one - third were against . However , the passage in question actually refers to the French Revolution of 1789 .
Robert D. Marcus ( 1971 ) . The American Scene : Varieties of American History . p. 147 . See also `` Only 1 / 3 of Americans Supported the American Revolution
? '' , by William Marina . 6 - 28 - 2004 . Retrieved on July 14 , 2008 . Jump up ^ Ray Raphael ( 2012 ) . A People 's History of the American Revolution . The
New Press . p. 393 .
</item 10>
=== END ITEM id=2c60a00d99e01305 ===

=== ITEM id=ebc3a6a8ddfa104c ===
Gold sentences and facts:
Sentence 0: Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 . A sequel , Deadpool 3 , is in development .
  Fact 0.0: Deadpool 2 is scheduled to be released in the United States on May 18, 2018.
  Fact 0.1: A sequel, Deadpool 3, is in development.

Retrieved items:
<item 1>
Deadpool 2
Josh Brolin emerged as a `` surprise contender '' to play Cable in April , ahead of Shannon and Harbour , and was officially cast in the role . Brolin also
portrays the Marvel Comics character Thanos , in the Marvel Cinematic Universe . Leslie Uggams confirmed that she would be reprising her role of Blind Al from
the first film , while Fox gave the sequel a June 1 , 2018 release date . Noting the release date , Leitch wanted to ensure that the film `` was worthy of a
summer tentpole movie , and we knew we were going to be wedged in between some big films '' , specifically wanting to expand the action and make the general
feeling of the film `` bigger '' than the first one . However , he wanted the film to have the same DNA as the original `` in terms of the tone , and the fun
... I love that challenge '' of combining that with the increased scope . The film uses Deadpool 's fourth wall - breaking to reference this release period ,
with Leitch calling these comments `` definitely fresh and timely when they come up . ''
</item 1>

<item 2>
Deadpool 2
During filming , stunt woman Joi `` SJ '' Harris died in a motorcycle accident . Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 .
A sequel , Deadpool 3 , is in development . Contents ( hide ) 1 Premise 2 Cast 3 Production 3.1 Development 3.2 Pre-production 3.3 Filming 3.4 Post-production 4
Music 5 Release 5.1 Marketing 6 Box office projection 7 Sequel 8 References 9 External links Premise Deadpool forms a team of mutants called the X-Force to
protect young mutant Russell from the time - traveling soldier Cable . Cast Ryan Reynolds as Wade Wilson / Deadpool : A wisecracking mercenary with accelerated
healing but severe scarring over his body after undergoing an experimental regenerative mutation . He forms the X-Force , a team of mutants . Josh Brolin as
Nathan Summers / Cable : A time travelling cybernetic mutant soldier , `` in many ways the opposite of Deadpool '' .
</item 2>

<item 3>
Deadpool 2
By April 26 , opening weekend industry projections had reached upwards of $150 million . Sequel By November 2016 , with development underway on Deadpool 2 , Fox
was also planning Deadpool 3 , which was said to include the team X-Force . With the confirmation that Leitch would direct Deadpool 2 , it was revealed that Fox
was looking for a separate filmmaker to direct Deadpool 3 . In March 2017 , Reese clarified that though Deadpool 2 sets - up the X-Force team , a future film
focused on the team would be separate from Deadpool 3 , `` so I think we 'll be able to take two paths . ( X-Force ) is where we 're launching something bigger
, but then ( Deadpool 3 is ) where we 're contracting and staying personal and small . '' After the proposed acquisition of 21st Century Fox by Disney was
announced in December 2017 , Disney CEO Bob Iger said that Deadpool would be integrated with the Marvel Cinematic Universe under Disney , and that the company
would be willing to make future R - rated Deadpool films `` as long as we let the audiences know what 's coming '' .
</item 3>

<item 4>
Graph query row:
movie: Deadpool 3
release_date: None
</item 4>

<item 5>
Deadpool 2
: Next Dimension X-Men Legends X-Men Legends II : Rise of Apocalypse X-Men : The Official Game X-Men : Destiny Wolverine Wolverine Wolverine : Adamantium Rage
X-Men : Wolverine 's Rage X2 : Wolverine 's Revenge X-Men Origins : Wolverine Characters in other media Apocalypse Cyclops Gambit Magneto Jean Grey Professor X
Psylocke Rogue Storm Wolverine See also X-Men storylines X-Men comics Live - action theatrical films based on Marvel Comics Franchises Ant - Man Ant - Man Ant -
Man and the Wasp Avengers The Avengers Avengers : Age of Ultron Avengers : Infinity War Untitled Avengers film production Blade Blade Blade II Blade : Trinity
Captain America Captain America ( serial ) Captain America ( 1990 ) Captain America : The First Avenger Captain America : The Winter Soldier Captain America :
Civil War Daredevil Daredevil Elektra Deadpool Deadpool Deadpool 2 Fantastic Four The Fantastic Four ( 1994 ) Fantastic Four ( 2005 ) Fantastic Four : Rise of
the Silver Surfer Fantastic Four ( 2015 ) Ghost Rider Ghost Rider Ghost Rider : Spirit of Vengeance Guardians of the Galaxy Guardians of the Galaxy Guardians of
the Galaxy Vol.
</item 5>

<item 6>
Deadpool 2
Jump up ^ Eisenberg , Eric ( March 21 , 2017 ) . `` Why Deadpool 3 And X-Force Will Be Very Different Projects '' . CinemaBlend . Archived from the original on
May 24 , 2017 . Retrieved May 24 , 2017 . Jump up ^ White , Peter ; Hayes , Dade ( December 14 , 2017 ) . `` Disney - Fox Deal : Bob Iger Discusses Digital
Future , James Murdoch , Hulu and $2 B Cost Savings '' . Deadline.com . Archived from the original on December 16 , 2017 . Retrieved December 16 , 2017 . Jump
up ^ Couch , Aaron ( December 14 , 2017 ) . `` ' Deadpool ' Can Stay R - Rated at Disney , Says Bob Iger '' . The Hollywood Reporter . Archived from the
original on December 16 , 2017 . Retrieved December 16 , 2017 .
</item 6>

<item 7>
Deadpool 2
Cable 's inclusion in the potential sequel was confirmed by the character Deadpool , breaking the fourth wall , in the post-credit scene of the first film .
Domino , a character with connections to Cable in the comics , was also believed to be featured in the sequel . By the first film 's release , 20th Century Fox
had green - lit a sequel , with writers Rhett Reese and Paul Wernick returning to write the screenplay . Though director Tim Miller and producer / star Ryan
Reynolds were not confirmed for the sequel at the time , Fox was `` intent on keeping the creative team together '' . Miller and Reynolds ' involvement was
confirmed at the 2016 CinemaCon that April , though Miller had still not formally signed on to direct the sequel yet . He began work developing the script with
the writers , while Reynolds had signed a new contract granting him `` casting approval and other creative controls '' . It has to tonally and stylistically be
as fresh and original ( as the first film ) . That 's a big challenge especially because they had 10 years to gestate on the first movie and we do n't have that
kind of time on the second movie .
</item 7>

<item 8>
Deadpool 2
Deadpool 2 - wikipedia Deadpool 2 Jump to : navigation , search Deadpool 2 Teaser poster Directed by David Leitch Produced by Simon Kinberg Ryan Reynolds Lauren
Shuler Donner Written by Rhett Reese Paul Wernick Ryan Reynolds Based on Deadpool by Rob Liefeld Fabian Nicieza Starring Ryan Reynolds Josh Brolin Morena
Baccarin Julian Dennison Zazie Beetz T.J. Miller Brianna Hildebrand Jack Kesy Stefan Kapičić Music by Tyler Bates Cinematography Jonathan Sela Edited by Craig
Alpert Elísabet Ronaldsdóttir Production company Marvel Entertainment The Donners ' Company Genre Films Distributed by 20th Century Fox Release date May 18 ,
2018 ( 2018 - 05 - 18 ) ( United States ) Country United States Language English Deadpool 2 is an upcoming American superhero film based on the Marvel Comics
character Deadpool , distributed by 20th Century Fox . It is intended to be the eleventh installment in the X-Men film series , and a sequel to the 2016 film
Deadpool .
</item 8>

<item 9>
Deadpool 2
Retrieved December 16 , 2017 . External links Official website Deadpool 2 on IMDb Deadpool Fabian Nicieza Rob Liefeld Supporting characters Agent X Avengers
Blind Al Bob , Agent of Hydra Sandi Brandenberg Cable Copycat Death Domino Emrys Killebrew Negasonic Teenage Warhead Outlaw Shiklah Siryn Spider - Man Weasel
Wolverine X-Men Antagonists Ajax Black Box Black Swan Black Talon Black Tom Cassidy Bullseye Doctor Bong Dracula Hit - Monkey Flag - Smasher Juggernaut Madcap
Sabretooth Slayback Taskmaster T - Ray Thanos Trapster Typhoid Mary Publications Cable & Deadpool Mrs. Deadpool and the Howling Commandos Wolverine and Deadpool
In other media Deadpool ( video game ) Deadpool ( film ) accolades box office records soundtrack No Good Deed Deadpool 2 Related articles 5 Ronin Gwenpool Mercs
for Money X-Force Deathstroke X-Men in other media Films X-Men X2 X-Men : The Last Stand soundtrack X-Men Origins : Wolverine X-Men : First Class The Wolverine
X-Men : Days of Future Past Deadpool soundtrack X-Men : Apocalypse Logan Deadpool 2 X-Men : Dark Phoenix The New Mutants Development of Gambit Characters
</item 9>

<item 10>
Deadpool 2
Reynolds , a fan of Leitch 's John Wick , said the director `` really understands those Deadpool sensibilities and where we need to take the franchise from here
. '' He added that Leitch `` can make a movie on an ultra tight minimal budget look like it was shot for 10 -- 15 times what it cost . '' Pre-production Reese
and Wernick had completed multiple drafts of the script by January 2017 , saying , `` It 's taken different twists and turns , but it 's really coalescing '' ;
the film was still on track to begin filming that year . The pair felt a responsibility to explore the team X-Force , which includes Deadpool , Cable , and
Domino in the comics , with Reese saying the sequel 's purpose `` is not to set up X-Force ( but ) it will likely set up X-Force . '' He did clarify that though
the film will be `` populated with a lot of characters ... it is still Deadpool 's movie '' .
</item 10>
=== END ITEM id=ebc3a6a8ddfa104c ===

=== ITEM id=c47aaa94391c7083 ===
Gold sentences and facts:
Sentence 0: The College Football Playoff selection committee chose the semifinalists following the conclusion of the 2017 regular season . Alabama and Georgia advanced to the national championship after winning the semifinal games hosted by the Sugar Bowl and the Rose Bowl , respectively , on January 1 , 2018 . The championship game was played at Mercedes - Benz Stadium in Atlanta , Georgia on January 8 , 2018 .
  Fact 0.0: The College Football Playoff selection committee chose the semifinalists.
  Fact 0.1: The semifinalists were chosen following the conclusion of the 2017 regular season.
  Fact 0.2: Alabama advanced to the national championship.
  Fact 0.3: Georgia advanced to the national championship.
  Fact 0.4: Alabama won its semifinal game hosted by the Sugar Bowl on January 1, 2018.
  Fact 0.5: Georgia won its semifinal game hosted by the Rose Bowl on January 1, 2018.
  Fact 0.6: The championship game was played at Mercedes-Benz Stadium in Atlanta, Georgia.
  Fact 0.7: The championship game was played on January 8, 2018.
Sentence 1: The Mercedes - Benz Stadium in Atlanta , Georgia was announced as the host site for the fourth College Football National Championship on November 4 , 2015 .
  Fact 1.0: Mercedes-Benz Stadium in Atlanta, Georgia was announced as the host site for the fourth College Football National Championship.
  Fact 1.1: The host site announcement was made on November 4, 2015.

Retrieved items:
<item 1>
2018 College Football Playoff National Championship
28 ) Camping World ( Dec. 28 ) Alamo ( Dec. 28 ) Holiday ( Dec. 28 ) Belk ( Dec. 29 ) Cotton ( Dec. 29 ) Sun ( Dec. 29 ) Music City ( Dec. 29 ) Arizona ( Dec.
29 ) Fiesta ( Dec. 30 ) Liberty ( Dec. 30 ) Orange ( Dec. 30 ) TaxSlayer ( Dec. 30 ) Citrus ( Jan. 1 ) Outback ( Jan. 1 ) Peach ( Jan. 1 ) College Football
Playoff games : Rose ( semifinal ) ( Jan. 1 ) Sugar ( semifinal ) ( Jan. 1 ) National Championship Game ( Jan. 8 ) All - Star Games East -- West Shrine Game (
Jan. 20 ) NFLPA Collegiate Bowl ( Jan. 20 ) Senior Bowl ( Jan. 27 ) NCAA Football Championship Games Division I FBS 2007 * 2008 * 2009 * 2010 * 2011 * 2012 *
2013 * 2014 * 2015 * 2016 * 2017 * 2018 * 2019 *
</item 1>

<item 2>
2018 College Football Playoff National Championship
Alabama and Georgia advanced to the national championship after winning the semifinal games hosted by the Sugar Bowl and the Rose Bowl , respectively , on
January 1 , 2018 . The championship game was played at Mercedes - Benz Stadium in Atlanta , Georgia on January 8 , 2018 . Contents ( hide ) 1 Background 2 2018
College Football Playoff 3 Teams 3.1 Alabama 3.2 Georgia 4 Game summary 4.1 Scoring summary 4.2 Statistics 5 Broadcasting 6 Dispute 7 See also 8 References 9
External links Background ( edit ) The Mercedes - Benz Stadium in Atlanta , Georgia was announced as the host site for the fourth College Football National
Championship on November 4 , 2015 .
</item 2>

<item 3>
The January 8, 2018, CFP National Championship featured Georgia versus Alabama; Kendrick Lamar performed a separate halftime show at Centennial Olympic Park,
while the stadium hosted the teams’ marching bands. Undefeated UCF claimed a national title after beating Auburn in the Peach Bowl, but the CFP committee had
ranked it 12th and excluded it from the playoff. Major polls did not recognize the claim. UCF planned a Disney World parade and a championship banner.
</item 3>

<item 4>
2018 College Football Playoff National Championship
related to 2018 College Football Playoff National Championship at Wikimedia Commons College Football Playoff Championships Overview College Football Playoff
National Championship Games 2014 -- 15 2015 -- 16 2016 -- 17 2017 -- 18 2018 -- 19 2019 -- 20 2020 -- 21 Semifinal Bowls Rose 2014 ( Jan 2015 ) 2017 ( Jan 2018
) 2020 ( Jan 2021 ) Sugar 2014 ( Jan 2015 ) 2017 ( Jan 2018 ) 2020 ( Jan 2021 ) Orange 2015 ( Dec 2015 ) 2018 ( Dec 2018 ) Cotton 2015 ( Dec 2015 ) 2018 ( Dec
2018 ) Peach 2016 ( Dec 2016 ) 2019 ( Dec 2019 ) Fiesta 2016 ( Dec 2016 ) 2019 ( Dec 2019 ) Other List of College Football Playoff games New Year 's Six College
Football Playoff National Championship Trophy NCAA Division I FBS champions Mythical national championship Plus - One system Power Five conferences Championship
games for each season are played in
</item 4>

<item 5>
The text is a fragmented index of college football championship games and seasons, listing years for the FBS/I-A, FCS/I-AA, Division II, and Division III
championships. It highlights the 2018 College Football Playoff National Championship, notes that an asterisk marks games played in January, and begins a list of
Alabama’s bowl games with Rose Bowl appearances in 1926, 1927, and 1931.
</item 5>

<item 6>
2018 College Football Playoff National Championship
Jump up ^ Alex Scarbrough ( November 4 , 2015 ) . `` Atlanta , Santa Clara and New Orleans land CFP title games for 2018 -- 20 '' . ESPN.com . Retrieved
November 4 , 2015 . ^ Jump up to : `` Selection Committee Rankings : Final Top 25 Rankings '' ( PDF ) . College Football Playoff . December 3 , 2017 . Retrieved
December 3 , 2017 . Jump up ^ `` Alabama 's Sugar Bowl win against Clemson sets up all - SEC final : Final score , stats , analysis '' . nola.com . January 1 ,
2018 . Retrieved January 2 , 2018 . Jump up ^ Emerson , Seth ( January 1 , 2018 ) . `` Georgia to play for national championship after Rose Bowl win in 2 OT ''
. ajc.com . Retrieved January 2 , 2018 . ^ Jump up to : Hoffman , Benjamin ; Drape , Joe ; Tracy , Marc ( January 8 , 2018 ) . `` National Championship Game :
Georgia vs.
</item 6>

<item 7>
2018 College Football Playoff National Championship
champions Mythical national championship Plus - One system Power Five conferences Championship games for each season are played in January , while semifinal
games are identified by season year , alternating between December and January Links to related articles 2017 -- 18 NCAA football bowl game season New Mexico (
Dec. 16 ) Las Vegas ( Dec. 16 ) Camellia ( Dec. 16 ) Cure ( Dec. 16 ) New Orleans ( Dec. 16 ) Boca Raton ( Dec. 19 ) Frisco ( Dec. 20 ) Gasparilla ( Dec. 21 )
Famous Idaho Potato ( Dec. 22 ) Bahamas ( Dec. 22 ) Armed Forces ( Dec. 23 ) Birmingham ( Dec. 23 ) Dollar General ( Dec. 23 ) Hawaii ( Dec. 24 ) Quick Lane (
Dec. 26 ) Heart of Dallas ( Dec. 26 ) Cactus ( Dec. 26 ) Independence ( Dec. 27 ) Pinstripe ( Dec. 27 ) Texas ( Dec. 27 ) Foster Farms ( Dec. 27 ) Military (
Dec. 28 ) Camping World ( Dec. 28 ) Alamo ( Dec.
</item 7>

<item 8>
2018 College Football Playoff National Championship
2018 College football Playoff national championship - wikipedia 2018 College football Playoff national championship Jump to : navigation , search 2018 College
Football Playoff National Championship Alabama Crimson Tide Georgia Bulldogs ( 12 -- 1 ) ( 13 -- 1 ) SEC SEC 26 23 Head coach : Nick Saban Head coach : Kirby
Smart AP Coaches CFP AP Coaches CFP OT Total Alabama 0 0 10 10 6 26 Georgia 0 13 7 0 23 Date January 8 , 2018 Season 2017 Stadium Mercedes - Benz Stadium
Location Atlanta , Georgia MVP Offensive : # 13 QB Tua Tagovailoa , Fr . Alabama Defensive : # 94 DT Da'Ron Payne , Jr .
</item 8>

<item 9>
2018 College Football Playoff National Championship
2018 College football Playoff ( edit ) Semifinals Championship January 1 -- Sugar Bowl Mercedes - Benz Superdome , New Orleans 1 Clemson 6 4 Alabama 24 January
8 -- National Championship Mercedes - Benz Stadium , Atlanta 4 Alabama ( OT ) 26 January 1 -- Rose Bowl Rose Bowl , Pasadena 3 Georgia 23 2 Oklahoma 48 3
Georgia ( 2OT ) 54 This bracket : view talk edit The College Football Playoff ( CFP ) selected four teams to advance to the playoff : the Clemson Tigers ,
Oklahoma Sooners , Georgia Bulldogs , and Alabama Crimson Tide . Teams ( edit ) Alabama ( edit ) See also : 2017 Alabama Crimson Tide football team The Alabama
Crimson Tide defeated the Clemson Tigers in the 2018 Sugar Bowl by a score of 24 -- 6 to reach the championship game . The Tide had a 12 -- 1 season , becoming
the SEC West Division co-champions with the Auburn Tigers . The Crimson Tide were coached by Nick Saban .
</item 9>

<item 10>
2018 College Football Playoff National Championship
1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018
Division III 1973 1976 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006 2007
2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 Games played in January are denoted by an asterisk Alabama Crimson Tide bowl games 1926 Rose Bowl 1927 Rose
Bowl 1931 Rose Bowl
</item 10>
=== END ITEM id=c47aaa94391c7083 ===

=== ITEM id=75c06ca2c637a92d ===
Gold sentences and facts:
Sentence 0: The soma ( cell body ) of each pseudounipolar neuron is located within a dorsal root ganglion . The axon leaves the cell body ( and out of the dorsal root ganglion ) into the dorsal root , where it splits into two branches . The central branch goes to the posterior ( dorsal ) horn of the spinal cord , where it forms synapses with other neurons . The peripheral branch travels through the distal dorsal root into the spinal nerve all the way until skin , joint , and muscle .
  Fact 0.0: The soma (cell body) of each pseudounipolar neuron is located within a dorsal root ganglion.
  Fact 0.1: The axon leaves the cell body.
  Fact 0.2: The axon leaves the dorsal root ganglion.
  Fact 0.3: The axon goes into the dorsal root.
  Fact 0.4: In the dorsal root, the axon splits into two branches.
  Fact 0.5: The central branch of the axon goes to the posterior (dorsal) horn of the spinal cord.
  Fact 0.6: The central branch forms synapses with other neurons.
  Fact 0.7: The peripheral branch travels through the distal dorsal root.
  Fact 0.8: The peripheral branch travels into the spinal nerve.
  Fact 0.9: The peripheral branch travels all the way until skin, joint, and muscle.

Retrieved items:
<item 1>
A pseudounipolar neuron is a sensory neuron of the peripheral nervous system with one branched axon rather than separate dendrites and an axon. Its peripheral
branch carries signals from receptors in the skin, joints, muscles, and other tissues; its central branch carries them toward the spinal cord. Cell bodies lie
in dorsal root ganglia, and central branches synapse in the spinal cord’s dorsal horn. Pseudounipolar neurons also occur in sensory ganglia of cranial nerves V,
</item 1>

<item 2>
Pseudounipolar neuron
Pseudounipolar neuron - wikipedia Pseudounipolar neuron Jump to : navigation , search Pseudounipolar neuron 1 : Pseudounipolar neuron , 2 : Bipolar neuron
Details System Nervous system Identifiers Latin neuron pseudounipolare TH H2. 00.06. 1.00047 Anatomical terms of neuroanatomy ( edit on Wikidata ) A
pseudounipolar neuron ( pseudo -- false , uni -- one ) is a kind of sensory neuron in the peripheral nervous system . This neuron contains an axon that has
split into two branches ; one branch runs to the periphery and the other to the spinal cord . Contents ( hide ) 1 Structure 1.1 In the dorsal root ganglia 1.2
In the sensory ganglia of cranial nerves 2 Function 2.1 Signal pathway 3 See also 4 References Structure ( edit ) By definition , a pseudounipolar neuron has
one axon with two branches : central and peripheral . These axonal branches should not be confused with dendrites .
</item 2>

<item 3>
Pseudounipolar neuron
These axonal branches should not be confused with dendrites . These sensory neurons are an exception to the typical neuron , in that they do not have separate
dendrites and an axonal process , but rather one branched process that serves both functions . The axon has a peripheral branch ( from the cell body to the
periphery : skin , joint and muscle ) and a central branch ( from the cell body to spinal cord ) . In the dorsal root ganglia ( edit ) The soma ( cell body ) of
each pseudounipolar neuron is located within a dorsal root ganglion . The axon leaves the cell body ( and out of the dorsal root ganglion ) into the dorsal root
, where it splits into two branches . The central branch goes to the posterior ( dorsal ) horn of the spinal cord , where it forms synapses with other neurons .
The peripheral branch travels through the distal dorsal root into the spinal nerve all the way until skin , joint , and muscle .
</item 3>

<item 4>
Pseudounipolar neuron
. Cambridge : McGrawHill . p. 24 . ISBN 978 - 0838577011 . ^ Jump up to : MacCallum , Don . `` Peripheral Nervous System '' . Histology and Virtual Microscopy
Learning Resources . University of Michigan Medical School . Archived from the original on 7 July 2014 . Retrieved 24 June 2014 . Jump up ^ Miller 's Anatomy of
the Dog , By Howard E. Evans , Alexander de Lahunta , 4th ed , pp 565 ( hide ) Nervous tissue CNS Tissue Types Grey matter White matter Projection fibers
Association fiber Commissural fiber Lemniscus Funiculus Fasciculus Nerve tract Decussation Commissure Neuropil Meninges Cell Types Neuronal Pyramidal Purkinje
Granule Glial insulating : Myelination : Oligodendrocyte other Astrocyte Radial glial cell Ependymal cells Tanycyte Microglia PNS General Dorsal Root Ganglion
Ramus Ventral Root Ramus Ramus communicans Gray White Autonomic ganglion (
</item 4>

<item 5>
Motor neuron
These columns are further explained in the table below . Motor Columns of Spinal Cord Motor Column Location in Spinal Cord Target Median Motor Column Present
entire length Axial muscles Hypaxial Motor Column Thoracic Region Body wall muscles Preganglionic Motor Column Thoracic Region Sympathetic ganglion Laterl Motor
Column Brachial and Lumbar region ( both regions are further divided into medial and lateral domains ) Muscles of the limbs Phrenic Motor Coulmn Cervical Region
Diaphragm Anatomy and Physiology ( edit ) Spinal cord tracts Location of lower motor neurons in spinal cord Upper motor neurons ( edit ) Upper motor neurons
originate in the motor cortex located in the precentral gyrus . The cells that make up the primary motor cortex are Betz cells , which are a type of pyramidal
cell . The axons of these cells descend from the cortex to form the corticospinal tract . White matter Tract ( edit ) White matter tracts are bundles of axons
and serve as means to propagate nerve impulses on a large scale . These are found in the spinal cord and appear white from the myelin surrounding axons .
</item 5>

<item 6>
Pseudounipolar neuron
In the sensory ganglia of cranial nerves ( edit ) Pseudounipolar neurons are also found in the sensory ganglia of cranial nerves V , VII , IX and X , all mixed
- cranial nerves . The vestibulocochlear nerve , has the bipolar neuron type in both the spiral ganglion as well as Scarpa 's ganglion . Function ( edit )
Signal pathway ( edit ) These neurons have sensory receptors on skin , joints , muscles , and other parts of the body . The area of the axon that is closest to
the receptor is the trigger zone for the neuron . The signal is conducted through the axon to the dorsal root ganglion 's cell body , then through the axon and
ending at the sensory nuclei in the posterior horn of the spinal cord . See also ( edit ) Bipolar neuron Multipolar neuron Unipolar neuron References ( edit ) ^
Jump up to : Kandel , Eric R. ; Schwartz , James H. ; Mack , Thomas M. Jessell ( 2000 ) . Principles of neural Science ( 4th ed . ) . Cambridge : McGrawHill .
p. 24 .
</item 6>

<item 7>
Upper motor neurons synapse mainly on spinal interneurons, sometimes directly on lower motor neurons. Lower motor neurons carry signals from the spinal cord to
effectors and include somatic, special visceral, and general visceral types. Somatic alpha, beta, and gamma neurons innervate skeletal muscle; alpha neurons,
whose cell bodies lie in the spinal ventral horn, activate force-generating extrafusal fibers. One neuron may connect to many fibers; together they form a motor
unit. Repeated signals
</item 7>

<item 8>
Motor neuron
These are found in the spinal cord and appear white from the myelin surrounding axons . While these tracts exist for both efferent and afferent nerves ,
efferent nerves will carry signals down the spinal cord toward their effector target . These tracts also serve as the place of origin for lower motor neurons .
There are seven major white matter motor tracts to be found in the spinal cord and are listed below : Lateral Corticospinal Tract Rubrospinal Tract Lateral
Reticulospinal Tract Vestibulospinal tract Medial Reticulospinal Tract Tectospinal Tract Anterior Corticospinal Tract Lower motor neurons ( edit ) Lower motor
neurons are those that originate in the spinal cord and directly or indirectly innervate effector targets . The target of these neurons varies , but in the
somatic nervous system the target will be some sort of muscle fiber . There are three primary categories lower motor neurons , which can be further divided in
sub-categories .
</item 8>

<item 9>
Spinal motor columns target distinct structures: median (axial muscles, entire cord), hypaxial (body wall) and preganglionic (sympathetic ganglia) columns in
thoracic cord; lateral columns (limb muscles) in brachial and lumbar regions; and the cervical phrenic column (diaphragm). Upper motor neurons arise in
precentral-gyrus Betz cells and descend via corticospinal tracts. Myelinated spinal white-m
</item 9>

<item 10>
Pseudounipolar neuron
Ganglion Ramus Ventral Root Ramus Ramus communicans Gray White Autonomic ganglion ( Preganglionic nerve fibers Postganglionic nerve fibers ) Connective tissues
Epineurium Perineurium Endoneurium Nerve fascicle Neuroglia Myelination : Schwann cell Neurilemma Myelin incisure Node of Ranvier Internodal segment Satellite
glial cell Neurons / nerve fibers Parts Soma Axon hillock Axon Telodendron Axon terminals Axoplasm Axolemma Neurofibril / neurofilament Dendrite Nissl body
Dendritic spine Apical dendrite / Basal dendrite Types Bipolar Unipolar Pseudounipolar Multipolar Interneuron Renshaw Afferent nerve fiber / Sensory neuron GSA
GVA SSA SVA fibers Ia or Aα Ib or Golgi or Aα II or Aβ and Aγ III or Aδ or fast pain IV or C or slow pain Efferent nerve fiber / Motor neuron GSE GVE SVE Upper
motor neuron Lower motor neuron α motorneuron β motorneuron γ motorneuron Termination Synapse Electrical synapse / Gap junction Chemical
</item 10>
=== END ITEM id=75c06ca2c637a92d ===
