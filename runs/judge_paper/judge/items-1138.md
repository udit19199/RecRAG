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

=== ITEM id=8c522f26750b215c ===
Gold sentences and facts:
Sentence 0: On Oct. 12, 2023, Israel warned 1.1 million Gaza residents in the northern section of the enclave to leave for the southern region, in advance of a potential ground invasion.
  Fact 0.0: On October 12, 2023, Israel warned 1.1 million Gaza residents to leave for the southern region.
  Fact 0.1: The warned Gaza residents were in the northern section of the enclave.
  Fact 0.2: Israel issued the warning in advance of a potential ground invasion.
Sentence 1: Another user said that no one can carry out such a major attack against Israel without intelligence support.
  Fact 1.0: Another user said that no one can carry out such a major attack against Israel without intelligence support.

Retrieved items:
<item 1>
Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas (TechCrunch, 2023-10-14T07:46:27+00:00)
Silicon Valley had taken an increasing interest in Palestine as a tech hub, but like the ecosystem itself, it’s nascent: To date, those working in the region
estimate that as much as $10 million has been invested in the Palestinian tech ecosystem.

Notably, in 2017, Salesforce founder and CEO Marc Benioff joined Silicon Valley luminaries in backing the first-ever coding academy to be created in Gaza.

Gaza Sky Geeks, an Alphabet-backed initiative based in Gaza that provides pre-seed investments, training and technology resources to Palestine’s Gazan
population, has been a beacon of entrepreneurship in the region.

All of that is now effectively gone, like the buildings in Gaza itself.

Israel is currently retaliating militarily against the attacks on its people, on its soil, and the hostages subsequently taken by Hamas — the ruling
organization in Gaza that kidnapped at least 150 people and took them into Gaza during brutal attacks on Israel last weekend that killed 1,300 people.
</item 1>

<item 2>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
With the fence as protection, manned deployment for Gaza border settlements was cancelled, (in any case, these had been reduced immediately after Operation
Protective Edge), units deployed in the area were reduced, replaced with cameras and other electronic means.

"When relying on technology, the expectation is that it will warn of the dangers and threats," Barnea says. "When the thinking is that you don't need soldiers
because there are monitors, you expect in advance that you will receive the warning about the threat via the technological means and no one imagines a situation
in which the observation array would be neutralized. On the contrary, it is seen as invulnerable. The problem begins when the enemy is also exposed to these
means: RCWS are great, they are important and cost millions, but they are visible to everyone, and everyone knows where they are. Once you begin believing that
the fence will stop all the relevant threats, you think even half a battalion on standby is enough.
</item 2>

<item 3>
Israel’s startup ecosystem: down but not out (TechCrunch, 2023-10-13T19:10:07+00:00)
Making sure everyone is OK

The founders and operators that we spoke to stressed after they ensured the safety of their teams, they have gingerly started to approach how to resume work.
It’s not a simple formula.

“Currently, the emphasis is on finding a balance between supporting Israeli team members, some of whom want to keep working to clear their minds a little from
what’s happening, and some who cannot even think of working,” Omer Davidi, CEO & co-founder, BeeHero, an Israeli agritech startup, told TechCrunch. “The
priority is to focus on core company operations, especially in the short term, as we await a clearer understanding of the unfolding situation,” Davidi said.

Yonatan Cohen, co-founder and CTO of Quantum Machines, says that the war is clearly affecting people, but they are returning to work. “I can’t deny that we all
had trouble concentrating on work during the first few days of the war, ” he said.
</item 3>

<item 4>
Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas (TechCrunch, 2023-10-14T07:46:27+00:00)
Last time I was in Gaza a year ago, almost everyone I talked to asked me how they could get a job and leave Gaza. They were worried about more bombings and
wanted to bring their children up in a place without such a high risk. These people just want to live normal lives.”

Those who are living in the West Bank say the activity in Gaza has had an inevitable impact.

“For a Palestinian young woman like me living in the West Bank, I can confirm that there has been a noticeable freeze in terms of activity,” said Leen Abubaker
of Flow Accelerator and co-founder at Sawaed19. “Tech companies are either operating on a very limited scale, with employees struggling to reach their offices
in the West Bank due to unsafe roads blocked by Israeli occupation forces and settlers, or they have been forced out of business entirely in Gaza.”
</item 4>

<item 5>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
Two weeks into the war, the Israel Money Laundering and Terror Financing Prohibition Authority (IMPA) received secret intelligence information from two European
countries that warned: "A large, well-known organization is raising funds for Hamas through posts on social networks, and fintech company platforms outside
Israel, under the guise of donations to Gaza residents." The information included the name of the organization, its fundraising methods, and the identity of the
entities behind it. Their concern was that, within days, the millions of dollars raised would go directly to the military arm of Hamas. In a swift joint
international action, at IMPA’s request, one of the European countries blocked the organization’s money pipeline by immediately freezing its financial
activities.

"We are talking about a matter of minutes here. From the moment the authorities in Europe received the information, and we requested the money be blocked, with
no court orders or delay, they blocked the transfer of millions of dollars to one of the most active organizations, and from it to Hamas.
</item 5>

<item 6>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
For security reasons, Israel has already told Chevron to shut down its offshore energy platform.

The Middle East crisis brings back memories of the Arab oil embargo of 1973, when OPEC cut oil to supporters of Israel. It used to account for 50 per cent of
the oil production back then. Although non-OPEC production has picked up and OPEC contribution is lower at 32 per cent, it is still sizeable to take oil prices
higher. A 10 per cent rise in crude price will result in global inflation being 0.4 percentage points higher next year.

The war may also impact Suez Canal trade, which is a key maritime route between Asia and Europe and accounts for 12 per cent of global trade by value and much
larger by volume; also 5 per cent of world oil trade.

As things stand today, trade may not yet be impacted through the Strait of Hormuz, which lies between Oman and Iran.
</item 6>

<item 7>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
It's amazing to see every day when we all assess the situation, how all these bodies put their egos aside and I'm proud to be a part of it. If any channel wants
an official spokesperson for the Israeli government, then we will be there because we cannot abandon the battlefield."

The world does not make life easy for Israel's spokespeople, to say the least, compared with the ease with which the Hamas narrative is adopted. "I remember an
interview the day after the bluff about the Al-Ahli hospital, which the Palestinians claimed had been hit by an Israeli missile, killing 800 civilians, but in
the end it turned out that it was actually by a splinter group of Islamic Jihad, and that no hospital was destroyed and certainly not 800 people were killed. I
explained to them in Sky News that Hamas lies and makes up numbers and the reporter asks me almost speechless with surprise, if I doubt Hamas' numbers."

Levy describes these moments as intensely frustrating.
</item 7>

<item 8>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
India shipped goods around $8billion to Israel. Petroleum products account for 65 per cent of the exports, followed by gems and jewellery (15 per cent) and
electrical machinery (3 per cent). Trade can be adversely impacted if operations at the Israeli ports are disrupted. Fertiliser movement from Israel’s Port of
Ashdod accounts for 3 per cent of global potash supply. Any disruption on this front will also be a dampener.

But, what if the current hostilities turn into a full-scale conflict?

What supports this proposition is the fact that while there have been many conflicts between Israel and Hamas in the past decades, there have been none of this
intensity. With at least 1,200 Israelis killed, this has been the biggest single-day killing of Jews since Holocaust, and Israel opposition is also united in
retaliating. Israel has termed it a ‘War’, formed a war management cabinet with a key opposition member, and vowed to change the look of the ‘Middle East’.
</item 8>

<item 9>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Thirdly, as long as the rockets are intercepted in the skies over Israel, and not the skies over Gaza, the Israeli home front is forced to experience sirens,
security rooms and shelters, and disrupted routine living. In other words, even the complete success of the Iron Dome left daily life on the home front in the
hands of the terrorist organizations."

In 2018, Ortal wrote about "the barrier" that, "it would be wrong to assume it will turn out to be more immune than its predecessors. The longer as we delay
cutting the direct connection between the Hamas rockets and those of our other enemies, who make extensive use of this method of operation, and the idea of
attacking our territory, the longer we will be condemned to watch from the sidelines and look at their constant enhancement."

A strategic surprise

In the months leading up to October 7, the former head of the Planning Division, Major General (Res.)
</item 9>

<item 10>
Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas (TechCrunch, 2023-10-14T07:46:27+00:00)
Entire middle-class neighborhoods are being destroyed.”

She said in the past if an entire neighborhood got destroyed, it was usually one bordering Israel and a poorer area, thus less impacting the tech sector.

“The tech sector is almost completely unable to function in Gaza right now,” she said. “Most people are in too much danger to be able to work; some have
evacuated three times in the past 24 hours, moving from friend’s house to family house, because each neighborhood they end up in is the next one being bombed.
They usually receive warnings to evacuate their homes 10 minutes before a bombing, so they don’t sleep and monitor the situation constantly ready to evacuate in
a minute’s notice.

“Most people have lost cell phone connections and internet access completely, or have some access to 2G only on their cell phones. Electricity is no longer
being provided even for a few hours a day now, and people are running out of gas for their generators,” she added.
</item 10>

<item 11>
'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change (Fortune,
2023-10-22T15:15:47+00:00)
Israel has suggested they could be moved to tent camps in the Sinai Peninsula — funded by the United Nations and the US — and then returned once military
operations end, said the people. It’s unclear if Israel has put the idea directly to Egypt.

“Clearly the hope on Israel and the US’s side had been that Egypt would accept economic incentives, at a time when it is suffering an economic crisis, to allow
Gazans into Egypt,” said Mirette Mabrouk, director of the Egypt program at the Washington-based Middle East Institute.

Yet “there is little to no constituency for going against decades of policy on not allowing Palestinian displacement,” she said, while the inducements “might
wind up becoming a political liability, especially in an election year.”

All the same, multiple economists, bankers and investors Bloomberg spoke with this month in Morocco during annual meetings of the International Monetary Fund
and the World Bank saw Egypt as likely to receive some economic backing, whatever its refugee stance.
</item 11>

<item 12>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
None of the authorities ever took the risk of relying on advanced technologies."

The fence that, until October 7, separated Israel and the Gaza Strip was completed in December 2021, 60 years after construction of the Berlin Wall began. It
was intended to protect against a major security threat, but in practice it was much leaner than that. The fence was almost unmanned by soldiers, and relied
mainly on sensors, above and below ground, surveillance cameras, and vehicle patrols. And, of course, there was no surrounding minefield.

After Operation Protective Edge in 2014 revealed that Hamas forces were able to penetrate the Gaza border settlements at several points via underground tunnels,
it was decided in 2016 to build a defense line focused on this threat called "the anti-tunnel barrier" that would replace the existing fence.
</item 12>

<item 13>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
On October 27, OFAC imposed a second round of sanctions on key Hamas-linked officials and financial networks. The statement read, "Today’s action targets
additional assets in Hamas’s investment portfolio and individuals who are facilitating sanctions evasion by Hamas-affiliated companies. The Treasury is also
designating a Hamas official in Iran and members of Iran’s Islamic Revolutionary Guard Corps (IRGC), as well as a Gaza-based entity that has served as a conduit
for illicit Iranian funds to Hamas and Palestinian Islamic Jihad (PIJ)."

The US is not the only one. Europe has also been mobilized in the economic fight against terrorism. "On the third day of the fighting, my colleagues from the
Netherlands and Germany, both veteran heads of their authorities for the prohibition of money laundering, phoned to let me know they wanted to establish an
international task force to assist Israel in the fight against the financing of Hezbollah, Islamic Jihad, and all terrorist organizations and their satellites.
</item 13>

<item 14>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
How does the method work?

"Money changers in places all over the world, let's say Turkey, receive money, and inform the hawaladars (money changers) in Gaza who transfer the money on
their behalf. When a customer comes to a hawaladar in Turkey and asks to transfer money to Gaza, he does not physically transfer money, but settles accounts
internally with money changers in Gaza, and Judea and Samaria. The funds are de facto not transferred. Occasionally, or when the amount of the debt reaches a
predetermined ceiling, a transfer is made between the money changers to reset the outstanding debt."

Is it possible detect the money pipeline between hawaladars and halt the transfer?

"Transfers are made between two ‘exchangers,’ and don’t include the identity of the beneficiaries, or the persons who gave the money. They don’t know what the
transactions were, they only know there are debts to offset. We are in a constant battle against this phenomenon.
</item 14>

<item 15>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
RELATED ARTICLES Tech experts battle to shut down Gaza financial pipeline

"Overnight, we started receiving hundreds of reports about unusual activity by customers of financial institutions in the State of Israel," she says. Before
October 7, IMPA received about 100 reports a day dealing with terror financing. That number jumped to about 1,000. "There is a 900% increase in reports on
terror financing, and we don't waste time, everything happens very, very quickly. As we receive the information, we simultaneously pass it on to the operational
professionals who locate the money pipelines, and block them. Some information will not necessarily lead to terror financing, but we don’t have time to waste.
Bank CEOs call and draw our attention to the information they convey to us that warrants special attention."

Has information from banks and financial institutions led to blocking the money pipeline for terrorism?

"There was very high-quality intelligence that we passed on to the security agencies. We don't always know the results."
</item 15>

<item 16>
Israel’s startup ecosystem: down but not out (TechCrunch, 2023-10-13T19:10:07+00:00)
Other countries and regions have reported similar declines in their private-market capital flows this year.

As the war enters its sixth day, many people are being called up to the reserves, or volunteering to help in other ways. With the tech industry skewing younger,
it’s estimated that anywhere between 10% and 30% of all of Israel’s tech employees are getting mobilized. More than 500 VCs have pledged their support of the
efforts, too.

“In the spirit of peace and unity, we encourage the global venture community to support and engage with Israeli startups, entrepreneurs, and investors as they
navigate through these challenging times,” the signatories wrote.

To better understand the on-the-ground situation for Israeli entrepreneurs, we spoke to a number of startup founders, PR people who work with startups and
investors. The predominant response was that the businesses have to continue to operate as best they can, to help ensure the economic viability of the country.
</item 16>

<item 17>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
In an article five years ago, Ortal, currently in active military service an unavailable for interview, wrote, "Israeli restraint as regards proactive thwarting
of offensive capabilities in the Gaza Strip stems from a clear Israeli strategy that strives to reduce the influence of the enemy on the current routine of life
in Israel, even at the price of worsening of the threat in the future." Ortal pointed out that this policy was natural and that "a Western nation cannot lead
its life in an endless state of emergency. But alongside the understanding of the need for a containment policy, it must be recognized that the risks are high,
as evidenced by the steeply worsening intensity of conflicts between Israel and the terrorist organizations in Gaza."

According to Ortal, in recent decades, Hamas and Hezbollah developed ballistic missiles and terror tunnels in response to Israeli policy that sought minimal
friction with the enemy through separation and withdrawal from the security zone in southern Lebanon, and the Gaza Strip.
</item 17>

<item 18>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
"Qatar transferred suitcases of cash to Gaza through Israel. Everyone knows this. This money allegedly went to pay salaries and civilian needs (education,
health, infrastructure), and therefore Israel transferred it, so it’s incorrect to say that it all went to fund terrorism."

From money-changers to chocolate

How have these billions reached the Gaza street, and Hamas over the years? There are quite a few methods, including cryptocurrencies and trade. But the main and
easiest method for money transfer to Gaza is hawala.

Ostrovitch-Levi explains, "Most of the money does not reach Gaza physically through the border crossings, but via hawala, a financial channel that is based on
trust and connections, and enables the transfer and exchange of funds between countries, without the cash physically passing through them. It’s a channel for
transferring the funds raised in other countries. In Gaza there are all sorts of wealthy individuals who can transfer large amounts in cash to Hamas upon
receiving request from entities abroad."

How does the method work?
</item 18>

<item 19>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
Do the explanations produce greater acceptance?

"Yes. There is understanding that we must end this war by destroying the capabilities of Hamas, otherwise we will be in the same situation in another six
months. This is something I explain a lot to the Irish, for example. The Irish have been pressing for a ceasefire for a long time and I explain to them that if
we had taken their advice and stopped firing then little Emily Hand would still be in captivity, and if we take their advice now to agree to a ceasefire then
Hamas will commit more atrocities, because it will feel empowered and it will come out of this war with a feeling that the world has its back.

"We are being asked difficult questions about the dead on the other side and we must remind them that everyone who was killed in the Gaza Strip over the last
month would be alive today if Hamas had not started the atrocities on October 7, and then decided to fight in populated areas, while preventing people from
leaving despite all of Israel's efforts for over a month to evacuate civilians from the battle zones."
</item 19>

<item 20>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
"The Israelis must be totally losing it right now."

the strategic planning that must have gone into this against one of the most high tech occupation and surveillance systems in the world will be studied by many,
the Israelis must be totally losing it right now https://t.co/oUX2rokEjh — Rania Khalek (@RaniaKhalek) October 7, 2023

Tarik Cyril Amar said: "And the intel, and the special forces training... both areas where Israel mythology will be very upset indeed."

And the intel, and the special forces training... both areas where Israel mythology will be very upset indeed. — Tarik Cyril Amar (@TarikCyrilAmar) October 7,
2023

Mjoachim, another user, said that Israel's defence forces have lost their fame. He said the IDF lost an army base, a general was captured right out of his
house.
</item 20>

<item 21>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
We don't always know the results."

"Gaza could be more beautiful than Dubai "

The network to block terror financing funds includes many entities, including private citizens. "The world's financial intelligence authorities are now at the
heart of this struggle, and we’re working together to thwart the financing channels that we’ve located. In addition, we’ve been approached by many experts from
the private sector; people with intelligence experience because they’ve worked in these organizations, or have high-tech experience, and have mobilized like
crazy. Israeli entrepreneurs in Silicon Valley have contacted us; they write code, search for information, and give us lots of information. We aggregate and
examine everything."

IMPA is charged with rooting out the sources of funding for terrorism, working together with the National Bureau for Counter Terror Financing of Israel (NBCTF)
in the Ministry of Defense, which is the coordinating body of all activities to thwart terror financing, the Shin Bet, Military Intelligence Directorate (AMAN),
and all other relevant intelligence and security agencies.
</item 21>

<item 22>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
Levy describes these moments as intensely frustrating. He recounts, "In a BBC interview, the interviewer told me 'Hamas denies what you are saying,' and you see
the frustration on my face when I tell her that of course Hamas denies this. Hamas denies that it is even holding hostages. Hamas massacres, beheads, rapes,
burns families, and then, day and night, denies that it did this.

"Some journalists around world see that Hamas continues to lie about the biggest things when the whole truth is in front of everyone's eyes, and they continue
to echo the lies. This is not only a kinetic war in Gaza, it is also an information war."

Two months now, Levy has been fighting in this arena with the tools at his disposal. "We are fighting to make it clear to the world that the information coming
out of Gaza comes either directly from Hamas or from independent reporters and photographers that are part of the system in Gaza.
</item 22>

<item 23>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Hamas' surprise attack on Israel has shocked many who are now questioning the country's famed intelligence agency Mossad, which is known for its
intelligence-gathering capability. For some, the belief that Mossad was among the best in the spying world was shattered today when Hamas carried out the
biggest coordinated attack in Israel and captured some territory in southern Israel. It is not the attack that has surprised people but the scale at which Hamas
operated and got control of some military bases and soldiers.

Also Read: Israel at war: Hamas cells, military compounds destroyed, US backs Tel Aviv; all you need to know

Younis Tirawi, who reports on Palestinian Affairs, said that Hamas militants had taken over Sderot police HQ and armored vehicles in the town. An analyst wrote
that some of the images coming out of Sderot over the last few hours were reaching degrees of horror "that I don't really think I can use words to properly
describe.
</item 23>

<item 24>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
The US had put stock in Israeli intelligence regarding Hamas in recent years and hadn’t monitored it independently.

The human factor

The technological concept is, of course, not limited to the border fence, or to intelligence, and is not just a product of Operation Protective Edge. For years,
the IDF has been praising the ethos of a small and smart army, implicitly if not explicitly. In recent years, it has publicized countless technological projects
and present itself as a high-tech army: establishing a technological division, called Shiloh, to coordinate the development of technologies for all branches of
the army, and with the Directorate of Defense Research & Development (IMOD DDR&D or MAFAT).
</item 24>

<item 25>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
At that time, the Ministry of Defense claimed it would be integrated into the "Smart and Lethal Border" project that was being tested along the northern border
of the Gaza Strip which, the Ministry of Defense stated, included mobile robots and military drones for carrying out defense missions, without endangering
soldiers' lives.

At the time of the completion of the fence, Prime Minister Benjamin Netanyahu, then in the opposition, referred mainly to the below-ground aspect, saying that
"Any time Hamas terrorists have tried to penetrate the border settlements through the tunnels, they were terminated underground. The below-ground barrier
surrounding the Gaza Strip has already saved many lives." Then-Minister of Defense Benny Gantz said, "The barrier, which is a superior initial technological and
inventive project that sets an iron wall, sensors and concrete between it and the residents of the south. It provides a sense of personal security that will
allow this beautiful region to continue to grow."
</item 25>

<item 26>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
"Discussions with countries around the world and immediate deterrence are not long-term solutions. Stopping the funds from Iran should be carried out at the
international level through FATF, through OFAC sanctions, through sanctions from the entire world. In the end, the solution to Iranian financing are global
sanctions. It’s not something that one country can prevent. The bottom line is that without money there is no terrorism, and no matter how hard they try to
decapitate the terrorist organizations, as long as they have economic oxygen, as long as the countries of the world do not fight terror financing properly,
these organizations will continue to breathe."

Published by Globes, Israel business news - en.globes.co.il - on November 19, 2023.

© Copyright of Globes Publisher Itonut (1983) Ltd., 2023.
</item 26>

<item 27>
Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar (Fortune,
2023-10-13T19:50:48+00:00)
While Israel has granted permits to about 17,000 Gaza residents to enter and work in Israel, the food, fuel and medical supplies that people in Gaza use all
first pass through Israel.

Israel controls two physical checkpoints along Gaza, which monitor both the entry and exit of people and trucks. Israel limits the kind and quantity of
materials that pass into Gaza. And the blockades generally prohibit Gazans who do not have work permits or special clearance – for medical purposes, for example
– from entering Israel.

Israel’s restrictions through the blockade intensified since Hamas’ surprise attack on 20 Israeli towns and several military bases on Oct. 7, with Israel then
announcing a broad blockade of imports into Gaza. This stopped all food, fuel and medical supplies from entering the region.
</item 27>

<item 28>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Ortal points out that between Operation Pillar of Defense in 2012 and Operation Protective Edge in 2014, Israel significantly reduced targeted countermeasures
against terrorist operatives, and none focused on Hamas operatives. According to his analysis, this was because, prior to the Iron Dome coming into use in 2011,
the most dangerous threat posed by Gaza developed in the period preceding Operation Protective Edge. The understandings that Israel reached with Hamas in 2012
were made "to stop the fighting", Ortal says, and to get life back to normal as quickly as possible. This, after air raid sirens sounded in Tel Aviv and
Jerusalem for the first time since the Yom Kippur War, and the education system shut down for more than a week.

Even after Iron Dome, Israel found it difficult to stand up against the significant disruption of daily life on the home front.
</item 28>

<item 29>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Truly disturbing acts of terror directed at civilians".

Some of the images coming out of Sderot over the last few hours are reaching degrees of horror that I dont really think I can use words to properly describe.
Truly disturbing acts of terror directed at civilians. — OSINTtechnical (@Osinttechnical) October 7, 2023

Commenting on the coordinated attack, Rania Khalek, a journalist, said that the strategic planning that must have gone into this against "one of the most
high-tech occupation and surveillance systems in the world" will be studied by many. "The Israelis must be totally losing it right now."
</item 29>

<item 30>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Matania explains that the IDF has always advocated technological advancement as part of a concept that values quality over quantity. "This existed even during
[Israel’s first Prime Minster David] Ben-Gurion’s time, because it was clear that we could not defeat the enemy in terms of quantity, and it also integrated
well with saving human lives - an important pillar in the security concept - but at that time, innovation was expressed in tactical issues. After the Yom Kippur
War, Israel pulled strongly in the direction of building military technological superiority, and this integrated well with the Israeli economy, which was
gradually opening up to the world, and the computer revolution.

This effort bore fruit as early as 1982, when we proved to the whole world that we could destroy most of the Syrian surface-to-air missile array using
precision-guided weaponry. Israel's security concept was updated by Dan Meridor and Lt. Col.
</item 30>

<item 31>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Technology greatly influences our way of thinking and is seen as the answer to everything - only no one ever asks themselves what the vulnerabilities are, and
what is the backup plan for the doomsday scenario when the technology layer collapses."

Barnea compares Israel’s sense of technological superiority with US activity during the Vietnam War between 1965 and 1968. "The Americans relied on massive
aerial bombardment with precision bombs and napalm bombs that burned huge areas, and advocated the "scorched earth" concept. They convinced themselves that the
Vietcong could not withstand it. In practice, the North Vietnamese army studied the American activity. It dug tunnels, moved equipment and people between
bombings, eventually enabling it to reach the important bases in the south of the country, and overwhelm the Americans. In fact, the US military so believed in
the damage it was causing that they reported a far greater number of casualties on the other side than the actual number, which in turn created an illusion of
victory.
</item 31>

<item 32>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
Alongside this, tourism, commerce, transport services, entertainment, and restaurants will be hurt. As in the Covid pandemic, the self-employed and small and
medium-size businesses may be hurt, particularly those that operate with high leverage."

On Monday, the Bank of Israel announced a dramatic plan to sell $30 billion of its foreign currency reserves to stabilize the shekel. The shekel has in fact
recovered slightly, and is traded at around NIS 3.96/$.

What’s your view of the Bank of Israel’s move in the foreign exchange market?

Leiderman: "It’s important to welcome the Bank of Israel’s decision to intervene, as necessary, in the foreign exchange market, in order to ensure the market’s
regular operation. The decision to extend the term of Amir Yaron as governor of the bank at least for the duration of the war, and I hope beyond that, is also a
stabilizing factor. The banking system is stable, and there is enough liquidity in the system.
</item 32>

<item 33>
Office rents in Israel stay firm - for now (Globes English | Israel Business Arena, 2023-12-03T15:15:32+00:00)
Less than a month before the war, "Globes" reported a huge deal in which Amnon Shashua's unicorn AI21 Labs announced it was leasing 7,000 square meters for NIS
15 million annually in Tel Aviv's Da Vinci project.

It will now be interesting to see when the next such big deal for office space is signed in Tel Aviv following the impact of the Israel-Hamas war. Sources in
the income-producing property sector in Israel say that over the past few weeks there has been some interest in the sector, although mainly for smaller deals
involving hundreds of square meters each.

The reports: "No change in rents

Azrieli Group (TASE: AZRG), is one of the biggest companies in the sector, with 15 office buildings comprising 632,000 square meters of space and 97% occupancy.
Last week Azrieli published its financial results for the third quarter of 2023 saying that the company had not been impacted by the war.
</item 33>

<item 34>
Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas (TechCrunch, 2023-10-14T07:46:27+00:00)
His co-founder, Raya Fatayer, is in Ramallah, staying at home with her baby and husband, not able to travel.

“Our fellow entrepreneurs in Gaza had their homes demolished by air strikes, some we can’t even reach anymore since the electricity is out and they have no
power,” he said. “Dealing with the situation while trying our best to move forward with our work is a daily challenge.”

He said this outbreak of hostilities with Israel is clearly different: “Before, every time Gaza faced air strikes, we knew that certain areas were almost safe.
Clearly today, nobody is safe.”

(This piece was later edited to include relevant information about the attacks by Hamas).
</item 34>

<item 35>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
On Saturday, October 7, for example, suspicious movement in Gaza convinced the Shin Bet head to send a special team to the southern border, but he was unable to
convince the top IDF echelons to prepare accordingly. To handle those weak signals, Eiland claims, one must use "scenario theory", meaning, taking into account
scenarios that have a low probability of occurring, but with devastating results should they occur, and the simple way to prepare for them.

Eiland explains: "You detect certain noises on Friday night, but conclude that the probability of an attack is low, let's say only 10%. Those who act only on
the probability do not consider the question: if this attack happens - will you be able to live with yourself for eliminating this threat? There’s also a third
consideration, which is ease of effort: what is the effort required of me to prepare for this unlikely but dangerous scenario?
</item 35>

<item 36>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Published by Globes, Israel business news - en.globes.co.il - on November 5, 2023.

© Copyright of Globes Publisher Itonut (1983) Ltd., 2023.
</item 36>

<item 37>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
We are in a constant battle against this phenomenon. In recent years, Israeli authorities have designated several currency service providers in Gaza as conduits
facilitating the transfer tens of millions of dollars a year to terrorist organizations in Gaza."

Another way to transfer money is trade. Transferring goods, or inflating their value. For example, only days ago, containers of goods worth hundreds of
thousands of shekels were seized at Ashdod Port, which were destined for Hamas merchants in Nablus and other cities, with the proceeds intended to go on to
finance Hamas operations. Another example is the 23 tons of chocolate bars that were seized in August 2021 in a joint operation by the NBCTF, AMAN, and the Tax
Authority at the Nitzana Border Crossing between Israel and Egypt. It was suspected that the snack bars, destined for Gaza, were purchased with terror financing
funds, with the proceeds from their sales earmarked for Hamas.

Another affair that was uncovered, combined trade and hawala.
</item 37>

<item 38>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
We have a great deal of information about loss of growth and product in various episodes in the past, such as the Yom Kippur War, the Lebanon War, two
intifadas, various military operations, and the Covid pandemic.

"We see this is a completely different episode, however. This is a new situation that has no precedent as far as the damage to the economy is concerned. There’s
a whole spectrum of possible scenarios, from a short war on the southern front only, to the nightmare scenario of a prolonged war with additional sectors
involved. We are only in the fifth day of the war, and it’s to be hoped that the picture will start to become clearer."

Which industries will be hit by the war?

"A substantial decline can be expected in private consumption, particularly of services. This is not just because of the damage to output and income, but also
for psychological reasons. Distress, uncertainty, the possibility of prolonged reserve duty by some of the public - all these things will affect consumer
sentiment.
</item 38>

<item 39>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
BHARAT is not immune to #terrorism and internal enemies @HMOIndia @PMOIndia @AmitShah — Dr. Jaswant Gandhi 🇮🇳 (@JaswantDr) October 7, 2023

However, some speculated that Mossad knew what was going to happen but let Hamas make a mistake so that Israel could launch a ferocious attack on it. "Mossad
was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, but they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful," a user said.

#Mossad was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful
</item 39>

<item 40>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
A massive strike by Israel in Gaza, potentially killing civilians in collateral damage, may attract more of the Middle East countries, including the de-facto
leader Saudi Arabia to raise serious issues against Israel on humanitarian grounds. It will surely slow down the deal between Saudi and Israel that the US was
facilitating. Besides Gaza, clashes have already been ongoing along Israel’s borders with Syria and Lebanon. If Iran gets dragged into the conflict, on account
of claims of funding Hamas, this could embroil the entire Middle East in crisis.

In this scenario, we see a major upward pressure on world inflation, on account of disruption in crude oil and world trade logistics.

Possible disruption in crude and trade logistics

Even before the Hamas attack, Saudi Arabia and Russia further cut oil production, in an already tight supply market. Iran is a big incremental supplier of oil,
contributing 3 per cent to the global supply and any escalation risks taking energy prices higher. For security reasons, Israel has already told Chevron to shut
down its offshore energy platform.
</item 40>

<item 41>
Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar (Fortune,
2023-10-13T19:50:48+00:00)
International aid groups now face the same problem in Gaza that local businesses and residents have encountered for about 16 years: a blockade that prevents
civilians and items, like medicine from easily moving into or out of the enclosed area, roughly 25 miles long. That 16-year blockade did not apply to the food
and fuel that groups brought in to Gaza.

Now, it does.

Gaza’s blockade and economy

Gaza is about the size of Philadelphia and requires trade with different businesses and countries in order to maintain and grow its economy.

But Gaza is heavily dependent on foreign aid.

This is partially the result of Israel setting up permanent air, land and sea blockades around Gaza in 2007, one year after Hamas rose to political power.
Egypt, which borders Gaza on its southern end, also oversees one checkpoint that specifically limits people coming and going.
</item 41>

<item 42>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
The world can ill-afford another full-fledged war, when economies are already battling high inflation and interest rates, an ongoing Russia-Ukraine war and now,
a slowing China. This is precisely why many countries are trying to restrict any escalation in the ongoing Israel-Hamas conflict. The non-escaltion of the
current hostilities seems to be the base case presumed by the Indian stock market presently, going by the market movement in the last week.

The US treasury secretary also stated that the current conflict is not likely to significantly impact global economic outlook. History too shows that the these
conflicts tend to have limited impact on markets. We studied the history of conflicts between Israel and Hamas, and found that these tend to have limited impact
on global financial markets, across crude oil, US 10-year yields, US equities and market volatility (see charts).

However, as far as trade goes, there may be some impact. In 2022-23, Israel’s share in exports from India stood at 1.87 per cent.
</item 42>

<item 43>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Giora Eiland, held a series of lectures for senior officers on the reasons for the strategic surprise that fell on the IDF in the Yom Kippur War. "Not one of
the officers imagined that 50 years after that mistake, it would repeat itself in exactly the same way," he tells Globes.

The strategic surprise that hit the State of Israel about four weeks ago bears similar characteristics to the Japanese attack on the US naval base at Pearl
Harbor in 1941; to the Argentine surprise attack on Britain's Falkland Islands in 1982; the Al-Qaeda attack on the United States on September 11, 2001; and to
the spread of ISIS throughout the Middle East in 2014.

In fact, Barnea claims, despite the technological progress and sophistication, intelligence officials are questioning whether there really is a trend towards
improvement of early detection and thwarting of strategic threats.
</item 43>

<item 44>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
The upper section of the barrier was based mainly on a fence called the "sand clock," which had only proved itself in stopping illegal immigrants on the
Israel-Egypt border - the same fence that then-US President Donald Trump enthused over, and wanted to duplicate on the US southern border.

The cost of the barriers was NIS 3.5 billion, mostly for the below-ground section. It was presented as a high-tech project and as a great technological
achievement that would be an additional layer of protection from the Gazan threat, along with the Iron Dome. The Ministry of Defense (MOD) boasted that "the
amount of concrete invested in it could pave a road from Israel to Bulgaria, and the amount of iron and steel could equal an iron bar from here to Australia".
</item 44>

<item 45>
Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas (TechCrunch, 2023-10-14T07:46:27+00:00)
Palestine’s growing tech industry has been literally blown apart by the war between Israel and Hamas

Gaza, despite being one of the most economically challenged regions in the world, has ironically always been a tech hub — not only for Palestine and
Palestinians, but also for the world: International companies have, for many years, sought out a presence there to collaborate with talented tech freelancers
and the startups that gradually emerged from the region. For example, according to sources who helped build those bridges, Nvidia, famed for its role in the new
AI boom, has been working with at least 100 engineers from the region for years. (Tragically, the initiative to employ Palestinians in this outsourcing was that
of Mellanox founder Eyal Waldman, whose daughter was brutally murdered in the attacks by Hamas).

Since at least 2008, TechCrunch been covering technology companies out of Palestine, some serving their direct audience, some serving the tech world
internationally.
</item 45>

<item 46>
Office rents in Israel stay firm - for now (Globes English | Israel Business Arena, 2023-12-03T15:15:32+00:00)
"The company's management estimates that given that the war will focus only on the southern front and the duration of the fighting at the current intensity will
not exceed two to three months, then the effect of the war is not material on the company's business. In the aforementioned situation, the company's management
estimates that despite the damage to its revenue, which cannot yet be accurately estimated, they will be reflected in the company's annual NOI and FFO within
the forecast. "In the company's assessment, the continuation of the fighting for a long time and/or a full conflict on the northern border front (or on other
fronts) will result in a significant and wider damage to the economy, including deepening the damage to private consumption and businesses, including the
company's tenants, and as a result will cause a decrease in payments and changes in other economic parameters."

The supply: Millions of square meters between Kfar Saba and Rishon Lezion

Even before the war, the industry was talking about there being too large a supply of office space.
</item 46>

<item 47>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Concerns about the conflict spreading more broadly have eased amid growing calls within Israel to rethink a ground invasion of Gaza because of fears about the
fate of some 200 hostages held there, the danger of retaliation by Hezbollah and the risk of Israeli military casualties. Tweet of the day Quote of the day “I
think when you get to those decisions, you can then sit on them for a while… or you can say let’s just deal with it [and] that’s what we have done today,” said
Magellan chairman Andrew Formica as the group’s chief executive stepped down abruptly amid the company’s efforts to turn around its performance.
</item 47>

<item 48>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Hamas is an excuse, new lands are wonderful



I definitely support Israel pic.twitter.com/AbLdeazQjR October 7, 2023

Israeli Prime Minister Benjamin Netanyahu said that since this morning, Israel had been at war and its first objective was to clear out the hostile forces that
infiltrated its territory. "The second objective, at the same time, is to exact an immense price from the enemy, within the Gaza Strip as well," he said, adding
that the third objective was to reinforce other fronts so that nobody should mistakenly join this war. "We are at war. In war, one needs to be level-headed. I
call on all citizens of Israel to unite in order to achieve our highest goal – victory in the war," Netanyahu said.

Since this morning, the State of Israel has been at war. Our first objective is to clear out the hostile forces that infiltrated our territory and restore the
security and quiet to the communities that have been attacked.
</item 48>

<item 49>
Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas (TechCrunch, 2023-10-14T07:46:27+00:00)
The UN has warned of “devastating humanitarian consequences” from these latest moves. A total blockade on the territory is being enforced with fuel, food and
water running out. Israel says it won’t lift the restrictions unless Hamas frees all hostages.

Speaking to Ryan Sturgill, an American national and former head of the Gaza Sky Geeks (GSG) accelerator run by sponsor Mercy Corps, an NGO aid organization, the
situation on the ground appears dire, after waves of shelling by the Israeli military.

“The area around the the Mercy Corps building, which housed Gaza Sky Geeks, has been leveled. The structure is standing but blown out. The front of it is sort
of ripped off,” he said.

Gaza Sky Geeks is the largest tech hub in Palestine, providing a wide range of tech training at scale. In 2022, 5,000 coders and developers from across the West
Bank and Gaza graduated from the program.
</item 49>

<item 50>
Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas (TechCrunch, 2023-10-14T07:46:27+00:00)
That strategy has seen Israel pummeling the “Gaza Strip” with bombs to eradicate it of Hamas and to get its hostages back. Over 1,500 people in Palestine so far
have been killed as a result. The tech industry in Israel — the country’s biggest export and its biggest single contributor to GDP — is also taking a big knock,
but inevitably the impact on the smaller and more fragile ecosystem in Gaza has been significantly more serious. The physical, economic and societal destruction
resulting from that leaves any future for the tech industry there in doubt.

Quite simply, there is no escaping the consequences of the war for anyone, let alone tech workers.

“What is happening to tech in Gaza is that Israel is crunching it. Obliterating it,” one source inside the territory told TechCrunch.

Israel has now amassed soldiers near the north of Gaza, ahead of an expected ground offensive into the densely populated enclave. About 1.1 million people
living in northern areas have been told to leave in the next day.
</item 50>

<item 51>
Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas (TechCrunch, 2023-10-14T07:46:27+00:00)
Sturgill said that as well as the intensely difficult conditions in Gaza, which is being hit by Israeli missiles, the situation in Ramallah is “super tense. I
feel like the situation is going to get significantly worse there over the coming weeks.”

Iliana Montauk, co-founder and CEO of Manara — a social impact startup funded by Y Combinator, Seedcamp, Reid Hoffman, Eric Ries, Marc Benioff, Paul Graham, and
Jessica Livingston, among others — told TechCrunch via email that connectivity has decreased significantly in the past 24 hours.

“Though Gaza has been bombed many times before, this time is completely different for the tech sector for several reasons. Electricity was cut off to the entire
[Gaza] strip. A significant amount of infrastructure has been bombed (including both ISPs and many tall apartment buildings that hold cell phone towers). Entire
middle-class neighborhoods are being destroyed.”
</item 51>

<item 52>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
It provides a sense of personal security that will allow this beautiful region to continue to grow."

"The concept underlying the construction of an advanced technology fence relates to the Second Lebanon War, when Hezbollah surmounted a high fence with a
ladder," says Yehoshua Kalisky , senior researcher at Tel Aviv University's Institute for National Security Studies (INSS). "This was the reference scenario.
There was no thought that they would use aircraft here to cross it or that they would come with a heavy machinery, and simply breach it."

The illusion of superiority

On the morning of Saturday, October 7, the above-ground section of the barrier turned out to be completely insubstantial, as was its fundamental concept of
defense. Using drones easily purchased on the Internet, Hamas disabled both the cameras transmitting images to control room monitors in real time, and the (See
and Fire) long distance stationary remote controlled weapon stations (RCWS).
</item 52>

<item 53>
Office rents in Israel stay firm - for now (Globes English | Israel Business Arena, 2023-12-03T15:15:32+00:00)
However the company added that due to the uncertainty about the length of the war and its development, it could not assess the future impact of the war on
business activities in Israel in general and on the company in particular.

Amot Investments (TASE: AMOT), another of the biggest companies in the income-producing property sector, has 1.15 million square meters of office space for
leasing with 93.5% occupancy with a further six projects in various stages of construction. Amot wrote, "In the third quarter several new contracts were signed
including the exercising of options and the renewal of immaterial contracts for 14,000 square meters of space for an unchanged rent."

Amot raises concerns about the continuation of the war which would lead to damage to the economy, including to the company's tenants. The report says, "The
uncertainty that existed in the profitable real estate market in Israel even before the outbreak of the war and that was felt in the moderation in demand and
the lengthening of the negotiations to close agreements, has intensified following the war.
</item 53>

<item 54>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Israel's security concept was updated by Dan Meridor and Lt. Col. Ron Eldadi, with the addition of a fourth component, defense, to the three components of
Israel’s traditional national security doctrine as determined by Ben-Gurion: deterrence, warning, and decisive victory, leading to the development of systems
such as the Iron Dome, and David's Sling (formerly called Magic Wand)."

But, emphasizes Matania, "The IDF senior officers aren’t technologically savvy as a whole, some see these systems as black boxes, and don't always understand
their advantages and disadvantages. At the West Point or Annapolis military academies, it's customary to teach technological subjects as well, because they
understand that the military profession today requires a basic understanding of the field. The correct way to integrate technology with security is to implement
it as part of an operational concept, not as a single component, and to assume that it is not perfect, but has vulnerabilities, and may suffer from system
failures.
</item 54>

<item 55>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
There had been a second line of defense, observation balloons, but for unknown reasons, all three had stopped working some weeks before the attack, and repairs
were postponed until "after the holidays."

Israel’s drone warfare fleet, which is capable of jamming signals of a hostile drone, and the attack drone fleet, known for warfare against incendiary balloons,
also did not work for unknown reasons, and dozens of Hamas drones operated without interruption. The IDF's Iron Beam high energy laser (HEL) interception
system, which was developed for such cases, among other things, has not yet entered into operational activity after many years of development. All allowed the
Hamas terrorists to quickly reach the IDF bases along the Gaza border, disable the information and communications technology (ICT) systems, neutralizing the
ability to call reinforcements without arousing much suspicion at IDF Kirya headquarters in Tel Aviv.

Above all there was a sense of security and belief in technology and the sophisticated barrier’s ability to prevent any form of intrusion.
</item 55>

<item 56>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
The hundreds of millions of dollars that Hamas receives have been accumulating for years. We had no possibility of seeing a spike in fundraising towards October
7."

"We understand the west is next"

IMPA is not alone in the fight against terror financing. Three days after the start of the war, Andrea Gacki, Director of the US Treasury Department's Financial
Crimes Enforcement Network (FinCEN) contacted Ostrovitch-Levi to request a meeting. The two met at virtually the same time President Joe Biden met with the
families of the hostages. "She put herself and the entity she heads, one of the most powerful financial intelligence agencies in the world, at our disposal. She
said, 'Whatever you want or need, just say the word'".

Gacki has kept her promise. "Since that day, we’ve received an unprecedented flow of financial intelligence from the US, unlike anything seen before.
</item 56>

<item 57>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
I find it hard to understand how a war like this landed on us in a country as powerful as Israel. But we have no choice but to win."

As for the effect of the war on economic activity, Leiderman says: "First of all, the duration and character of the war will have a decisive impact on the depth
of damage to the economy. Many studies, as well as past experience, indicate that the longer the war lasts, the broader will be the impact on the economy. It is
also known that this connection between the length of the fighting and the economic damage might not be linear. And of course, the more the war widens to more
fronts, the greater will be the economic damage."

Do you have a forecast for the consequences of the war for economic growth?

"It’s too early to make a quantitative estimate of the effects on the growth of the economy in 2024 and 2025.
</item 57>

<item 58>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
Dozens of our staff have been called up to reserve duty, in large part managers working regularly in the fight against terror financing."

Since then, most of IMPA’s work has focused on identifying and blocking the terror money pipelines. "We work very reactively - we see a fundraising campaign,
try to deter it, and stop it from happening again. There are no breaks, because they are constantly active. It’s a daily struggle. Our analysts have been
sitting on the social networks from day two of the war, just manually monitoring visible sources of information on these networks to locate fundraising
campaigns for terrorist organizations operating in Gaza."

Three days after the war began, IMPA also issued a call to all global regulated entities for increased vigilance towards terrorism financing campaigns in
response to the ongoing war and state of emergency, and to report all activities that raise suspicion of terror support and terror financing, in an effective
and immediate manner.
</item 58>

<item 59>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
"How come they never knew Hamas militants planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate heavily? We wait to see how it goes
but it will be messy."

Mossad is overated how comes they never knew this Hamas militants and had planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate
heavily?We wait to see how it goes but it will be messy. pic.twitter.com/W5vpoylus2 — Ben Muasya. (@muasya_ben) October 7, 2023

Another user said that no one can carry out such a major attack against Israel without intelligence support. "Who gave Hamas Intelligence? It was as if the
entire Israeli army and the Mossad were asleep."
</item 59>

<item 60>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
It’s also possible to make bank transfers to Judea and Samaria, and the fundraising activity hasn't stopped, because while the fighting is going on, they’re
continuing to fill their reserves for ‘the day after’ [the war]. We realize they’re raising funds for their continued operations. Hamas is taking advantage of
the current sympathy for the Gazans’ situation, and is using online platforms, both of organizations whose business is crowdfunding, and also through the
creation of fictitious projects to raise funds: throw-pillows, jewelry and whatnot, and more."

All this happens openly, for all to see, in Europe, and other places around the world. "Hamas and Islamic Jihad carry out recruitment operations under the guise
of humanitarian aid campaigns, and use Telegram, Twitter, Facebook, Instagram and TikTok to disseminate their announcements. The facade is always supporting
humanitarian goals, but in practice it's about channels for funding individuals and entities connected directly to Hamas or Islamic Jihad."
</item 60>

<item 61>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
However, IMPA’s goal, Ostrovitch-Levi explains, is not only to stop the individual raising of funds for terror financing, but the entire interrelated chain,
from the fund-raiser to the transferrer, and on through to the recipient. "We have the ability to build the intelligence scenario, synchronize all the sources
of information received, and delineate the money trail from abroad to Hamas. We’re interested not only in who is running the campaign, the association or
entity, but where the money is coming from, and where it’s going. We need to stop all of the elements in this network."

Ever since October 7, everyone has been discussing and analyzing the question of about the intelligence agencies, and why there was no warning. Where was IMPA
before?

"Hamas collects lots of money all the time, and the security agencies deter money transfers. There was no sudden increase in the raising of billions, and you
must also remember that most of the money is raised globally, not in Israel.
</item 61>

<item 62>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
"I was asked twice, in Ireland and by the Voice of America, why Hamas had been begging for five weeks to return hostages and we were only now bowing to US
pressure to take back our children. I was interviewed by Piers Morgan in London who is very confrontational, who spoke to me about pictures from Gaza and said
'this the most shocking thing I have ever seen.' And I told him Piers, 'the most shocking thing we have seen in our lives was the slaughter on October 7.' Then
he answered 'you are right.'"

From questions like these, do you get the feeling that everyone in the world hates us and they are all anti-Semites?

"It's not true that everyone hates us and they are all anti-Semites. We enjoy a great deal of support in the world. People understand that what happened on
October 7 crossed a line with atrocities that cannot be repeated and for which there can be no excuse.
</item 62>

<item 63>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
They told themselves, we have B-52 bombers, there's no way we can't win."

Technological sophistication and the massive investment in intelligence means have greatly increased the IDF's visual intelligence gathering capabilities,
according to former IDF Intelligence Directorate head General (ret.) Aharon Ze'evi-Farkash. "I'm sure it doesn't come at the expense of human sources, but as
time goes by, gathering visual intelligence is easier than human sources and they bring better results. It's hard to bring in a Palestinian 'Ashraf Marwan,' the
agent who passed the information to Israel before the Egyptian attack in 1973. It’s a problem because the human source is essential to provide an interpretation
of the rest of the information that’s received, and to make decisions. Golda knew who Marwan was and expected to hear what he would say about the Egyptian
deployment in the south." At the same time, according to "The New York Times" a year before the attack, the IDF stopped monitoring Hamas' radio communications
last year.
</item 63>

<item 64>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Shiloh, which is currently subordinate to the Ministry of Defense, was established following State Comptroller reports about the lack of coordination and
synchronization between the Ministry of Defense and the IDF units; and lead a new border defense concept based on sensors and other technologies for gathering
intelligence and thwarting infiltration; strengthening the Air Force in the form of additional aircraft and armaments; and strengthening the operational end of
the regular front units.

All this, while at the same time closing down armored battalions and consolidating helicopter squadrons. Ground force technological capabilities were also
addressed, such as improved coordination between the infantry and air forces in air-to-ground actions, or identifying terrorists and weapons in an urban
environment. However, there is a problem at the outset, says, Prof. Eviatar Matania of Tel Aviv University School of Political Sciences, Government and
International Affairs, and founder and first director general of the Israel National Cyber Directorate: a large gap between the technology and the officers
expected to implement it as part of the security concept.
</item 64>

<item 65>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
With them and with the US, we established a kind of task force, which is not subject to any country or unifying body, countries that are working in
coordination, and meet every few days to exchange intelligence, and advance the struggle."

In general, following an appeal by IMPA to its counterparts around the world, there has been a dramatic increase in intelligence coming to Israel from abroad.
"The amount of information doubled in the month of fighting, compared with the monthly average. Before the war, we received about 24 intelligence reports a
month from parallel authorities, compared with 50 today."

Countries around the world cooperating with Israel to fight terrorism sounds almost ideal, but it’s hard not to wonder where they were before.

"Many countries felt this had nothing to do with them, that it wasn’t their struggle.
</item 65>

<item 66>
Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar (Fortune,
2023-10-13T19:50:48+00:00)
This stopped all food, fuel and medical supplies from entering the region.

Gaza’s isolation

The Palestinian enclaves of West Bank and Gaza – which are generally lumped together in economic analyses – both have small economies that run a massive deficit
of US$6.6 billion in losses each year, as the value of the imports they receive greatly outweighs the value of the items they produce and sell elsewhere.

More than 53% of Gaza residents were considered below the poverty line in 2020, and about 77% of Gazan households receive some form of aid from the United
Nations and other groups, mostly in the form of cash or food.

Gaza’s weak economy is caused by a number of complex factors, but the largest is the blockade and the economic and trade isolation it creates.

For the average Gazan, the blockade has several practical effects, including people’s ability to get food.
</item 66>

<item 67>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
The banking system is stable, and there is enough liquidity in the system. The behavior of the investment community in Israel, including the financial
institutions, indicates high confidence in the economy’s resilience and financial strength. It’s important to maintain a level-headed, correct economic policy
that will preserve this ‘asset.’"

What about the economic cost of the war?

"As with any war, the current war has and will have direct and indirect impacts. The war will substantially raise defense spending in the coming years, for
procurement of equipment, ammunition, and advanced systems in accordance with the new needs. Another direct expense is payment for the reserve duty days of
those mobilized. Given current interest rate levels around the world, the cost of servicing public debt as a proportion of GDP will rise. Besides all this,
there will be a decline in state revenues from taxation and a rise in the fiscal deficit."

How can this larger deficit be financed?
</item 67>

<item 68>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
"Despite the clear threat posed by Gaza, the decision makers preferred to avoid a policy that would mean returning a large part of the country's population to
the reality of air raids and disrupted daily routines. The bottom line: the Iron Dome did not change the essential strategic situation; as firepower from the
Gaza Strip developed, so did the terrorist organizations’ ability to control daily life in Israel’s hinterland, hold it hostage, and restrain Israel from taking
proactive measures against them."

"Why didn't the relative success of the Iron Dome translate into the even more important strategic achievement of restoring Israel's freedom of action against
Hamas in the Strip?" asks Ortal rhetorically, and names several factors. "The first, like any other operational system, Iron Dome is also not perfect. The enemy
studies it and develops its own challenges against it, and the decision makers recognized the fact that the State of Israel is not hermetically protected.
Second, the enormous cost involved in deploying and operating [Iron Dome] batteries significantly impedes a policy based on continuous rounds of violence.
</item 68>

<item 69>
Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar (Fortune,
2023-10-13T19:50:48+00:00)
International aid groups are warning that they cannot deliver food and other basic services to people in the Gaza Strip and that a “dire” humanitarian crisis is
set to worsen.

International aid groups provide food and other means of support to about 63% of people in Gaza.

Israel stopped allowing deliveries of food, fuel and other supplies to Gaza’s 2.3 million residents on Oct. 10, 2023, and is reportedly preparing for a ground
invasion.

On Oct. 12, 2023, Israel warned 1.1 million Gaza residents in the northern section of the enclave to leave for the southern region, in advance of a potential
ground invasion.

I am a scholar of peace and conflict economics and a former World Bank consultant, including during the 2014 war between Hamas and Israel.
</item 69>

<item 70>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
Although the situation remains precarious, connectivity eventually was partially restored. There are indications that the U.S. government is putting pressure on
Israel to restore internet connectivity in Gaza.

Meanwhile in India, Apple has notified several prominent Indian lawmakers and journalists of potential state-sponsored attacks on their iPhones, just months
ahead of the country’s general elections. The individuals targeted include opposition leader Rahul Gandhi, key Congress party figure Shashi Tharoor, and leaders
from several other influential political parties.

Atlassian issued a warning about a critical security flaw that could result in significant data loss for its customers. Notably, this product was recently the
target of Chinese state-sponsored hackers. Atlassian has not yet detailed how the flaw can lead to data loss but has urged customers to patch against the flaw
immediately.

There’s more:

About flippin’ time, too: Apple has recently fixed a longstanding vulnerability in its iOS software that undermined a privacy feature. Security researchers
found that the feature, introduced in iOS 14, was not working as intended.
</item 70>

<item 71>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
Urgent letter to @elonmusk on #DSA obligations ⤵️ pic.twitter.com/avMm1LHq54 — Thierry Breton (@ThierryBreton) October 10, 2023

Graphic videos apparently showing terrorist attacks on civilians have been circulating on X since Saturday, along with other content, including some posts that
purport to show footage from the attacks inside Israel or Israel’s subsequent retaliation on targets in the Gaza Strip but which fact-checkers have identified
as false.

The Hamas attacks on Israeli civilians and tourists, which took place after militants inside Gaza managed to get past border fences and mount a series of
surprise attacks, have been followed by Israel’s prime minister declaring “we are at war” and its military retaliating by firing scores of missiles into the
Gaza Strip.
</item 71>

<item 72>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
He said the IDF lost an army base, a general was captured right out of his house. "Invincible "Merkava MK4 MBT's captured, and destroyed...what is wrong with
the Israeli security and defence forces? And (what) MOSSAD did?"

Dr Jaswant Gandhi, a professor of management, said that attack on Israel was a "great Mossad failure". He said India had to learn a lot and keep all
intelligence agencies duly greased and internecine conflict-proof. "Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence
Agencies duly greased and internecine conflict-proof. BHARAT is not immune to #terrorism and internal enemies," he wrote on X.

#Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence Agencies duly greased and internecine conflict proof.
</item 72>

<item 73>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
Most of the money comes from oil, but there’s actually an Iranian security system budget dedicated to this."

Another problem to be dealt with is the addiction to the cash received from the terrorist organizations. "Because it involves cash in such huge amounts, a
situation has arisen in which the populations in Gaza and Lebanon become addicted to financing from Iran. There is internal collapse and a complete disruption
of the ability to receive funds another way. This makes terrorism a very profitable business. It destroys the civic structure, unemployment rates rise very
high, and being a Hamas operator becomes profitable. It's a vicious circle that feeds on itself. This is an alternative banking system disconnected from the
general system. The Iranians are building whole mechanisms to enable intake of this money. This is something bigger than deterring one transfer or another."

Can the current global mobilization alter the fight against terror financing?

"Discussions with countries around the world and immediate deterrence are not long-term solutions.
</item 73>

<item 74>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
"The most significant way to finance terrorism is to finance countries, not fundraising campaigns. These have intensified greatly over the last month, but the
core of the budget - for Hamas, Hezbollah, the Houthis, and the Palestinian Islamic Jihad is Iran, in particular the Quds Force (the special forces of the IRGC,
Iran's Islamic Revolutionary Guard Corps). Hamas has all kinds of additional sources of self-financing, including businesses, associations, investments, and all
kinds of arrangements they’ve tried to set up over the years. They need state funding, and without Iran they have no ability to exist."

How much money does Iran transfer to Hamas?

"As far as we know, the budget from Iran to Gaza is in the order of $100 million a year. Gaza could do very nice things with this money, make it even more
beautiful than Dubai. But from what we know, most of this budget goes to the military wing of Hamas."

Qatar also transfers money to Hamas.

"Qatar transferred suitcases of cash to Gaza through Israel.
</item 74>

<item 75>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
They’ve given us valuable information that we’ve passed on to the security agencies, about organizations they didn’t necessarily know about, and people backing
them, all in real time. At the same time, they’ve issued alerts to increase vigilance among the US public about terror financing."

On October 18, the US Treasury Department's Office of Foreign Assets Control (OFAC) announced that it had imposed sanctions on 10 key Hamas operatives and
financial facilitators in Gaza and elsewhere including Sudan, Turkey, Algeria, and Qatar. The statement read, "This action targets members managing assets in a
secret Hamas investment portfolio, a Qatar-based financial facilitator with close ties to the Iranian regime, a key Hamas commander, and a Gaza-based virtual
currency exchange and its operator… To date, the Treasury has targeted nearly 1,000 individuals and entities connected to terrorism and terrorist financing by
the Iranian regime and its proxies, including Hamas, Hezbollah, and other Iran-aligned terrorist groups in the region."
</item 75>

<item 76>
Office rents in Israel stay firm - for now (Globes English | Israel Business Arena, 2023-12-03T15:15:32+00:00)
Naimi says that the first building to be occupied, with an area of 35,000 square meters, "Is mostly marketed and there is already set to be occupancy." The
first company to move in as early as January is Audicodes, which has leased 10,500 square meters and is implementing finishing works. "In March-April,
additional companies will enter. Sheba Medical Center Tel Hashomer will also enter there and leased 2,600 square meters and will increase to 5,300 square
meters," he says.

Ben Zvi Klein thinks that as in other crises, when the crisis breaks everything halts but then we see buds of everything returning to work. But he also adds
that volumes are low but no different than before the war.

Ben Zvi Klein says, ""In the first two weeks of the war, people were in the market but the market was at a standstill and once again people started to wonder,
what will happen now and where are we going and how long will it take.
</item 76>

<item 77>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
It may be a relatively simple effort, like waking the troops up at 4:00 am, placing them position, and putting an aircraft in the air for a few hours. You don’t
even need pilots - unmanned aircraft operators are enough. There’s no need to mobilize reserves or convene a cabinet. Every commander is allowed to give
instructions to those under him - if that would have happened, everything would be different.

"This is exactly the same type of failure that was at the basis of Israeli thinking at the outbreak of the Yom Kippur War," Eiland continues. "After information
was gathered that indicated the intentions of the Egyptians and Syrians to launch an attack, they decided not to mobilize reserves because the cost was high.
But why didn't they order the regular army to get ready? They thought the chance of war breaking out was low, they thought it was an exercise, but they didn't
calculate the potential damage, they didn't imagine how serious the extreme scenario could be."
</item 77>

<item 78>
Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar (Fortune,
2023-10-13T19:50:48+00:00)
This article is republished from The Conversation under a Creative Commons license. Read the original article.
</item 78>

<item 79>
Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar (Fortune,
2023-10-13T19:50:48+00:00)
About 64% of people in Gaza are considered food insecure, meaning they do not have reliable access to sufficient amounts of food.

Food as a percentage of Gaza’s total imports has skyrocketed by 50% since 2005, when Israel first imposed a temporary blockade. And the amount of food the West
Bank and Gaza actually produce has tumbled by 30% since then.

It is hard for Gaza to produce food within its own borders. One factor is that Israeli airstrikes hit Gaza’s only power generation plant and main sewage
treatment plant in 2008 and again in 2018. These attacks resulted in the spread of sewage waste on land and in the water, destroying farmlands and food crops
and threatening fish stocks in the ocean as well.

The UN’s big role in Gaza

Gaza’s weak economy and isolation because of the blockade mean that it relies heavily on international aid organizations to provide basic services to residents.
</item 79>

<item 80>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
A comprehensive study conducted on intelligence failures in the national arena, from the fall of the Iron Curtain in 1989 to 2010, showed no improvement in
providing or thwarting warnings about strategic surprises, even if the means of gathering information had improved. Four years after signing the study, former
US President Barack Obama admitted that US intelligence had not correctly assessed the severity of the threat of large parts of Iraq and Syria being occupied by
ISIS.

"No matter how technologically advanced your army or how effective your intelligence gathering system, identifying strategic surprises is a human process that
depends on analyzing what is called, in the intelligence community, 'weak signals,'" Barnea says. "These are pieces of relevant information that have been
conveyed to the organization, but because they are low-key, or because of the information overload around them, the organization fails to identify and interpret
them correctly."
</item 80>

<item 81>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
The first week of fighting in the bitter war in Israel’s south is almost over, with the financial markets trying to gauge the damage that the Israeli economy
will sustain. Clearly, we are talking about a very significant event, but the fog of war makes it hard to estimate its full consequences. The main index on the
Tel Aviv Stock Exchange, the Tel Aviv 35, fell 6.4% over the week.

Prof. Leo Leiderman, chief economic adviser at Bank Hapoalim and a professor of economics at Tel Aviv University, sets out for "Globes" his proposals for
rehabilitating the economy and repairing the damage that the war will cause.

What has been going through your mind in these first days of the war?

"What most preoccupies me is not the economy but concern for the families of the soldiers, the wounded, and the captives, and solidarity with them. I have four
relatives from Kibbutz Nir Oz who are still missing, and we have no information about them.
</item 81>

<item 82>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
may again get impacted on account of higher input costs and higher freight and fuel costs.

Rural India is a key engine that is not firing at the moment. Valuations also are rich and do not offer margin of safety in the near to medium term. Near-term
attractiveness of equities will reduce, as investors will demand a higher risk premium. The Indian stock market will likely correct materially, as FIIs look to
withdraw funds from risk assets and seek safe havens in US Dollar, US treasuries offering high yields and Gold.

Sectors which will be negatively impacted include Petrochemicals, Oil Marketing, Paints, Packaging, Textiles and Chemicals.

The writer is Group President & Head - Institutional Equities, YES Securities India

SHARE Copy link

Email

Facebook

Twitter

Telegram

LinkedIn

WhatsApp

Reddit
</item 82>

<item 83>
Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar (Fortune,
2023-10-13T19:50:48+00:00)
Some Republican politicians have said that UNRWA has “cozied up” to Hamas. And an internal UNRWA ethics committee has accused top staff at the agency of “sexual
misconduct, nepotism, retaliation … and other abuses of authority” that created a toxic work environment.

Meanwhile, since the war between Israel and Hamas began on Oct. 8, more than 1,500 Gazans have been killed and more than 5,300 injured, while Hamas attacks have
killed more than 1,300 people in Israel and injured about 3,200 others.

International aid groups and European Union officials have called for a humanitarian corridor to be set up in Gaza – meaning a protected path specifically for
civilians, aid workers and necessary basic items to pass through safely back and forth from Gaza to Israel and Egypt. So far, there are no clear plans for such
a protected pathway.

Topher L. McDougal is Professor of Economic Development & Peacebuilding, University of San Diego.
</item 83>

<item 84>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
"Many countries felt this had nothing to do with them, that it wasn’t their struggle. The international and US mobilization, the task force we founded, the fact
that the financial sector abroad suddenly wants to accept the Israeli announcements about the terror entities and people involved, receive intelligence from
Israel, and cooperate with us is all only because of October 7. Everyone understands that ‘The West is next’ isn’t an empty statement. This event is the one to
make the countries of the world understand they must fight effectively against terror financing, that they need to block Iran effectively."

IMPA is trying to convey these messages to the member nations of the Financial Action Task Force (FATF), an international body to combat money laundering and
terror financing. "We are working very hard on isolating Iran in matters of terror financing. But it’s still a cash economy and it is very, very difficult to
stop the financing of Hamas, Hezbollah, and Islamic Jihad.
</item 84>

<item 85>
Israel's prince of public diplomacy (Globes English | Israel Business Arena, 2023-12-03T12:50:26+00:00)
Everyone knows that even if they are not affiliated with Hamas, they are not free to present the full truth, because they are not allowed to. There is not a
single photo of a terrorist, there is not a single photo of a missile being launched from a densely populated area, from a mosque or a school, not even a single
photo from the Hamas tunnels under Shifa. Why? Because these reporters, even if they are not explicitly on the side of Hamas, operate under a dictatorial regime
and cannot provide the information even if they wanted to. And it is very important to continue reminding the world how much the information that reaches them
is biased and does not reflect reality."

Despite all this, Levy feels he must stress that not everyone is against us. "I don't want to give the impression that all the interviewers are hostile and
every question is delusional. Absolutely not. I have many interviews with reporters who understand what is happening here and simply want to get information and
hear the Israeli side, which is completely legitimate."
</item 85>

<item 86>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Iron Dome was so successful it not only supported the decision-making process but fundamentally changed it. Up until October 7, it served as a very successful
plaster to threats to a sovereign state, because what legitimacy would a state leader have for going to war when no citizens were harmed?"

Brigadier General Eran Ortal, until recently the commander of The Dado Center for Interdisciplinary Military Studies, a military research center subordinate to
the Operations Directorate, also believes that various technological measures such as the Iron Dome, and the barrier, influenced decision-making in Israel. In
his estimation, these helped the political echelon conduct a policy of military restraint towards the Gaza Strip, which in turn allowed Hamas to strengthen and
gain offensive capabilities.
</item 86>

<item 87>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
This is a real-time deterrence," reveals IMPA director general Adv. Ilit Ostrovitch-Levi. "And then there was a domino effect. Following this blocking, more
countries began to look around, and blocked funds intended for terror. All this information, which is now shared at an international level, leads to the closing
of more and more Hamas funding pipelines, in real time. With worldwide cooperation, Israel’s security and enforcement authorities are thwarting the transfer of
millions of dollars earmarked for terrorism, every day."

Headed by Ostrovitch-Levi since May 2022, IMPA is an intelligence body that conducts the most complex of international economic investigations, and fights
terror financing routinely. On October 7, Ostrovitch-Levi received a painful reminder of the vital importance of her work. "Already on that Saturday night, with
Israeli under heavy rocket fire, we talked about preparing ourselves for the campaign we’re now entering, the fight against terror financing channels.
</item 87>

<item 88>
Israel’s startup ecosystem: down but not out (TechCrunch, 2023-10-13T19:10:07+00:00)
“There is no doubt that in the long run — a year or more — Israel’s tech ecosystem will emerge from this war, stronger, bigger, with more investment and returns
for investors,” he said. “Israel’s tech ecosystem has well proven its resilience in past conflicts with Hamas. Israel is focused on deep tech, semiconductors,
AI, mobility, fintech, cybersecurity, quantum, agtech, the cloud, devops, clean energy and so much more and as the future unrolls, these areas will grow in
importance and Israel will continue to play a leading role in all of them.”

Oren Yunger, managing partner at GGV, says his firm will do what it takes to make sure that the Israeli startups in his company’s portfolio don’t go bankrupt
due to the war.

“I can say with absolute confidence — even if we don’t know how long it’s going to take and who’s going to be affected — I am 100% sure that none of our
companies will go bust as a result from this moment in time,” he said.
</item 88>

<item 89>
Israel’s startup ecosystem: down but not out (TechCrunch, 2023-10-13T19:10:07+00:00)
“As we come out of the initial shock and horror of the attack, many tech workers and executives see it as their duty to keep the tech industry up and running.
Technology is Israel’s biggest export and the main engine of our economy and with a long and difficult war ahead, it’s crucial that we keep working and
innovating,” he told TechCrunch.

Sharon Seeman, a partner at YL Ventures, says that it’s important that companies maintain communication with customers. “Ensuring the continuity of business
operations is vital for Israeli startups in maintaining their customer relationships,” she said.

What might be a more tricky situation will be when startups come around to the inevitable cadence of their existence: raising money to scale their operations,
and then doubling down and scaling those operations. Should the region and country continue to be at war, how will that play out with would-be startup investors
and the LPs of startup investors?

OurCrowd founder and CEO Jon Medved believes that in the long run, the Israeli startup ecosystem will remain reslient.
</item 89>

<item 90>
Office rents in Israel stay firm - for now (Globes English | Israel Business Arena, 2023-12-03T15:15:32+00:00)
Companies began to check every investment. Along with this, interest rates rose and inflation is rampant all over the world. We are not a bubble. Above all
this, comes the war. However, in general, real estate works in waves and the wave will come again."

The war: "I expected a longer stoppage"

Natalie Marshall, the owner of Marshall Strategic Real Estate marketing talks about a restored interest in offices. "I expected there to be a longer stoppage in
the office market. There was a stoppage of office marketing for a month, but about two weeks ago movement recommenced. There is also seasonality in the office
sector - in July-August and during the holidays every year there is a stoppage.

"In my opinion, in an ironic way, all of us in the public have now realized that we have nowhere to go and we only have one country. In the last six months,
with the protests and the reform, people did not know what to do and even thought of leaving.
</item 90>

<item 91>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
The Berlin Wall was built in 1961 as a standard wire and concrete fence to make it difficult for East German residents to escape to freedom in the West. By the
end of the 1970s, its fourth generation already included a 127 km long electrified fence, a strip of land made of sharp metal nails, a closely-packed row of
watchtowers, a patrol road, an intrusion-tracking dirt road, guard dogs, deep trenches anti-vehicle trenches, and two layers of concrete walls. "The fence was
initially quite weak, so it was decided to place guards every few meters on top of the wall," Dr. Avner Barnea, a former Shin Bet senior officer and lecturer on
intelligence and national security, and business intelligence at Bar-Ilan University, tells "Globes." "But until they mined the area, they didn't really succeed
in preventing escapees. These low-tech solutions made all the difference. None of the authorities ever took the risk of relying on advanced technologies."
</item 91>

<item 92>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
The second objective, at the same time, is to exact an… pic.twitter.com/MzKs7tfv4M — Benjamin Netanyahu - בנימין נתניהו (@netanyahu) October 7, 2023

Reacting to this, however, a user said that knowing the proficiency of the Mossad he found the idea of a "surprise attack" to be a willing lie.

Well, you got the war you wanted Beni...at what cost of Israeli lives you don't care. Knowing the proficiency of the Mossad I find the idea of a "surprise
attack"to be a willing lie on your side of this. https://t.co/35qRvgBJt2 — timm (@timmainsson) October 7, 2023

Ben Muasya, a journalist, said that Mossad is overrated.
</item 92>

<item 93>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
It’s important to talk to the private sector, and to understand in every place and every industry what the main damage is and what they need in order to get
back to normal. It should be done professionally, preferably with no politicization of the recommendations."

What do you think the Bank of Israel should do in its next interest rate decision, on October 23?

"It’s still to early to tell how the war will affect the rate of inflation in the short and medium term. Just as there are new inflationary factors, such as the
depreciation of the shekel, there will also be factors working in the opposite direction, given the expected decline in private consumption, tourism, and other
items.

"As with fiscal policy, here too a country at war is allowed to deviate temporarily from the set inflation target. At the moment there is no surge in inflation
expectations, and the investment community is aware of the existence of index-linked assets and the exchange rate that can assist in hedging inflation risk.

"As for cutting the interest rate, it’s too early to consider that.
</item 93>

<item 94>
Israel’s startup ecosystem: down but not out (TechCrunch, 2023-10-13T19:10:07+00:00)
“This has inevitably strained many businesses,” he said.

Lazer Cohen, whose company Concrete Media works with many startups, says people are contributing however they can, but the businesses need to keep running, too.

“Looking at our clients and my agency, 10-25% of Israeli team members have been called up to reserve duty,” he said. Others on the team are helping by donating
blood, hosting survivors of the attacks, packing supplies for soldiers or covering for a team member who was called up.

The economy matters, too

While everyone is concerned about the war and its fallout, startup founders and their investors also recognize the importance of keeping these businesses going.
For most Israeli startups, the customer base is global and for many, their go-to-market teams are in the U.S. and hence not directly affected by the war. The
current situation may slow down product development but won’t directly affect sales in the short run.

While customers have been understanding for the most part, Cohen says that companies are working to continue operating under difficult conditions.
</item 94>

<item 95>
Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas (TechCrunch, 2023-10-14T07:46:27+00:00)
Many of my friends lost their houses in the Rimal area. There is no wifi internet, we connect to the cellular data on our phones but it’s only 2G and it
connects for a few minutes then cuts off. We can only send Whatsapp messages. So we can’t really read news on social media. In the morning we saw these videos
from our friends who shared it on whatsapp but it takes FOREVER to download a video of a few seconds.”

Montauk said Awad had not responded to her in the past day.

Montauk, a former director of Gaza Sky Geeks, said: “Before this escalation, the Gaza tech scene was growing. I was just in Riyadh and met companies that hire
entire software development teams in Gaza. Upwork and other Silicon Valley companies are now hiring software engineers remotely from Gaza. In addition, some had
left to work abroad at companies like Google, Amazon, Qualtrics, etc.
</item 95>

<item 96>
Office rents in Israel stay firm - for now (Globes English | Israel Business Arena, 2023-12-03T15:15:32+00:00)
It reminded me of the feeling during Covid, when there were ghost offices and nothing was happening. As time goes by, people come back to the offices. They
really want to leave home and get to work. Most offices have emergency shelters, and relative to residential areas, most of them are protected. I live in Tel
Aviv and during the fighting, when schools were closed, my children came with me to the office, because there is a shelter there and the house only has a
stairwell.

"In the last three weeks, we feel that the market is returning to activity - but we entered this war when the office market was already stuttering. There were
no huge deals, because there are no huge hiring in high-tech or big deals of tens of thousands of meters for the big companies. There are small and medium-sized
deals. There are quite a few inquiries from high-tech and service companies looking for 1,000-2,000 square meters who want to expand.
</item 96>

<item 97>
Israel’s startup ecosystem: down but not out (TechCrunch, 2023-10-13T19:10:07+00:00)
The world is still coming to terms with Hamas’ deadly attacks on Israelis last weekend — and everything else that has unfolded so far in the aftermath,
including the barrage of retaliation strikes on Gaza.

Undoubtedly, the ramifications for the technology ecosystem pale in comparison to the human toll that the fighting is taking, on both sides. But when
considering how the wider region will weather the destruction, and one day recover, it’s an important industry to consider. Please also read here about how this
is playing out for Palestinians in Gaza and beyond. And a below, we show how a look at how the situation is playing out in the startup ecosystem in Israel sheds
light on its impact to the country as a whole.

In such a small country, just about everyone knows someone who was directly affected by the attacks, or someone involved in the subsequent defense and
retaliation — and often all three.

Technology is, without question, a giant part of Israel’s economy.
</item 97>

<item 98>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
Therefore, such systems always need what’s known as 'redundancy', i.e. backup systems, to make sure there isn’t a single point of vulnerability through which
everything could collapse. The Air Force, for example, has several different models of aircraft just for this. Every plane also has several systems to back it
up."

The Iron Dome paradox

Perhaps more than anything else, the Iron Dome system is representative as the technological response to the threat from Gaza. Matania explains, "The Iron Dome
was built with the strategic vision of giving decision-makers breathing room, to allow them not to react immediately to the rockets, but to have the flexibility
to react at the right time, and in the right place, in a range of ways. But some say its success was paradoxical. Iron Dome was so successful it not only
supported the decision-making process but fundamentally changed it.
</item 98>

<item 99>
Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas (TechCrunch, 2023-10-14T07:46:27+00:00)
She added that a number of buildings in Gaza that are key to the tech industry there, such as Burj Al-Watan, had been destroyed by Israeli airstrikes and that
the tech industry is not the first priority for people in the urgent situation. “How can you possibly detach from the distressing reality and cling to the
remaining shreds of hope for your business?”

Mohammad Alnobani is a Palestinian founder of The Middle Frame, an Arab stock image platform powered by AI tools that aims to shatter stereotypes about the Arab
world through images and reducing bias in AI.

He told me he was on his way back from the One Young World Summit in Belfast, speaking about peace and reconciliation, and about to reach the borders to cross
to Palestine to get back to his family, when the war broke out.

“The borders closed down and I had to turn around and go back to Jordan,” he said. “I am still there, continuously checking on my family in Jerusalem, and
trying to reach out to my connections in Gaza.”
</item 99>

<item 100>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
Another affair that was uncovered, combined trade and hawala. Hamas funds originating in Iran, which were transferred in cash to money changers in Turkey for
transfer to Gaza. At the same time Hamed al-Khachari, a Gazan money changer who worked for Hamas (who was assassinated by Israel in 2019), located Gazan
merchants who imported goods from Turkish companies and owned payment on the goods. They paid cash to Khachari, who transferred the money to Hamas in Gaza.
Meanwhile, the money changers in Turkey received funds from Hamas abroad that were used to pay the Turkish companies.

Hamas is already preparing for "the day after"

Although it has been almost impossible to transfer money and goods to Gaza since October 7, with hawala shuttered, and no trade, Hamas continues to raise funds.
"The situation in Gaza is known, and nothing can be brought in, but fundraising activity continues in Judea and Samaria.
</item 100>
=== END ITEM id=8c522f26750b215c ===
